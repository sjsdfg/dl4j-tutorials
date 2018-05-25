package objectdetection;


import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Point;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.records.metadata.RecordMetaDataImageURI;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.util.ClassPathResource;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.datavec.image.recordreader.objdetect.impl.VocLabelProvider;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.TinyYOLO;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.List;
import java.util.Random;

import static org.bytedeco.javacpp.opencv_core.CV_8U;
import static org.bytedeco.javacpp.opencv_core.FONT_HERSHEY_DUPLEX;
import static org.bytedeco.javacpp.opencv_imgproc.*;

/**
 * from: https://gist.github.com/saudet/fb8a4d9544dc3c411b302ccd6bbf87e7
 *
 * Example transfer learning from a Tiny YOLO model pretrained on ImageNet and Pascal VOC to perform object detection with bounding boxes on images of red blood
 * cells.
 * <p>
 * References: <br>
 * - YOLO: Real-Time Object Detection: https://pjreddie.com/darknet/yolo/ <br>
 * - Images of red blood cells: https://github.com/cosmicad/dataset <br>
 * <p>
 * Please note, cuDNN should be used to obtain reasonable performance: https://deeplearning4j.org/cudnn
 *
 * @author saudet
 */
public class RedBloodCellDetection {
    private static final Logger log = LoggerFactory.getLogger(RedBloodCellDetection.class);

    public static void main(String[] args) throws Exception {
        detection();
    }

    public static void detection() throws Exception {
        // parameters matching the pretrained TinyYOLO model
        int width = 416;
        int height = 416;
        int nChannels = 3;
        int gridWidth = 13;
        int gridHeight = 13;

        // number classes for the red blood cells (RBC)
        int nClasses = 1;

        // parameters for the Yolo2OutputLayer
        int nBoxes = 5;
        double lambdaNoObj = 0.5;
        double lambdaCoord = 5.0;
        double[][] priorBoxes = { { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 } };
        double detectionThreshold = 0.3;

        // parameters for the training phase
        int batchSize = 2;
        int nEpochs = 50;
        double learningRate = 1e-3;
        double lrMomentum = 0.9;

        int seed = 123;
        Random rng = new Random(seed);

        String dataDir = new ClassPathResource("/datasets").getFile().getPath();
        File imageDir = new File(dataDir, "JPEGImages");

        // # fileMemory is not use for IntPointer, or for yolo example.
        // long fileMemory = 4 * 1000 * 1000 * 1000l;
        // log.info("WorkspaceConfiguration... fileMemory=" + fileMemory);
        // WorkspaceConfiguration mmap = WorkspaceConfiguration.builder().initialSize(fileMemory).policyLocation(LocationPolicy.MMAP).build();
        // try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(mmap, "M2")) {

        log.info("Load data...");

        RandomPathFilter pathFilter = new RandomPathFilter(rng) {
            @Override
            protected boolean accept(String name) {
                name = name.replace("/JPEGImages/", "/Annotations/").replace(".jpg", ".xml");
                try {
                    return new File(new URI(name)).exists();
                } catch (URISyntaxException ex) {
                    throw new RuntimeException(ex);
                }
            }
        };
        InputSplit[] data = new FileSplit(imageDir, NativeImageLoader.ALLOWED_FORMATS, rng).sample(pathFilter, 0.8, 0.2);
        InputSplit trainData = data[0];
        InputSplit testData = data[1];

        ObjectDetectionRecordReader recordReaderTrain = new ObjectDetectionRecordReader(height, width, nChannels, gridHeight, gridWidth,
            new VocLabelProvider(dataDir));
        recordReaderTrain.initialize(trainData);

        ObjectDetectionRecordReader recordReaderTest = new ObjectDetectionRecordReader(height, width, nChannels, gridHeight, gridWidth,
            new VocLabelProvider(dataDir));
        recordReaderTest.initialize(testData);

        // ObjectDetectionRecordReader performs regression, so we need to specify it here
        RecordReaderDataSetIterator train = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, 1, 1, true);
        train.setPreProcessor(new ImagePreProcessingScaler(0, 1));

        RecordReaderDataSetIterator test = new RecordReaderDataSetIterator(recordReaderTest, 1, 1, 1, true);
        test.setPreProcessor(new ImagePreProcessingScaler(0, 1));

        ComputationGraph model;
        String modelFilename = "model_rbc.zip";

        if (new File(modelFilename).exists()) {
            log.info("Load model...");

            model = ModelSerializer.restoreComputationGraph(modelFilename);
        } else {
            log.info("Build model...");

            ComputationGraph pretrained = (ComputationGraph)TinyYOLO.builder().build().initPretrained();
            INDArray priors = Nd4j.create(priorBoxes);

            FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder().seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(1.0).updater(new Adam.Builder().learningRate(learningRate).build())
                .updater(new Nesterovs.Builder().learningRate(learningRate).momentum(lrMomentum).build()).activation(Activation.IDENTITY)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED).inferenceWorkspaceMode(WorkspaceMode.ENABLED).build();

            model = new TransferLearning.GraphBuilder(pretrained).fineTuneConfiguration(fineTuneConf).removeVertexKeepConnections("conv2d_9")
                .addLayer("convolution2d_9",
                    new ConvolutionLayer.Builder(1, 1).nIn(1024).nOut(nBoxes * (5 + nClasses)).stride(1, 1).convolutionMode(ConvolutionMode.Same)
                        .weightInit(WeightInit.UNIFORM).hasBias(false).activation(Activation.IDENTITY).build(),
                    "leaky_re_lu_8")
                .addLayer("outputs", new Yolo2OutputLayer.Builder().lambbaNoObj(lambdaNoObj).lambdaCoord(lambdaCoord).boundingBoxPriors(priors).build(),
                    "convolution2d_9")
                .setOutputs("outputs")
                .build();

            System.out.println(model.summary(InputType.convolutional(height, width, nChannels)));

            log.info("Train model...");

            model.setListeners(new ScoreIterationListener(1));
            for (int i = 0; i < nEpochs; i++) {
                train.reset();
                while (train.hasNext()) {
                    model.fit(train.next());
                }
                log.info("*** Completed epoch {} ***", i);
            }
            ModelSerializer.writeModel(model, modelFilename, true);
        }

        // visualize results on the test set
        NativeImageLoader imageLoader = new NativeImageLoader();
        CanvasFrame frame = new CanvasFrame("RedBloodCellDetection");
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout = (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer) model.getOutputLayer(0);
        List<String> labels = train.getLabels();
        test.setCollectMetaData(true);
        while (test.hasNext() && frame.isVisible()) {
            org.nd4j.linalg.dataset.DataSet ds = test.next();
            RecordMetaDataImageURI metadata = (RecordMetaDataImageURI) ds.getExampleMetaData().get(0);
            INDArray features = ds.getFeatures();
            INDArray results = model.outputSingle(features);
            List<DetectedObject> objs = yout.getPredictedObjects(results, detectionThreshold);
            File file = new File(metadata.getURI());
            log.info(file.getName() + ": " + objs);

            Mat mat = imageLoader.asMat(features);
            Mat convertedMat = new Mat();
            mat.convertTo(convertedMat, CV_8U, 255, 0);
            int w = metadata.getOrigW() * 2;
            int h = metadata.getOrigH() * 2;
            Mat image = new Mat();
            resize(convertedMat, image, new Size(w, h));
            for (DetectedObject obj : objs) {
                double[] xy1 = obj.getTopLeftXY();
                double[] xy2 = obj.getBottomRightXY();
                String label = labels.get(obj.getPredictedClass());
                int x1 = (int) Math.round(w * xy1[0] / gridWidth);
                int y1 = (int) Math.round(h * xy1[1] / gridHeight);
                int x2 = (int) Math.round(w * xy2[0] / gridWidth);
                int y2 = (int) Math.round(h * xy2[1] / gridHeight);
                rectangle(image, new Point(x1, y1), new Point(x2, y2), Scalar.RED);
                putText(image, label, new Point(x1 + 2, y2 - 2), FONT_HERSHEY_DUPLEX, 1, Scalar.GREEN);
            }
            frame.setTitle(new File(metadata.getURI()).getName() + " - RedBloodCellDetection");
            frame.setCanvasSize(w, h);
            frame.showImage(converter.convert(image));
            frame.waitKey();
        }
        frame.dispose();
    }
}
