package baidudianshi;

import org.deeplearning4j.earlystopping.scorecalc.ScoreCalculator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Created by Joe on 2018/1/17
 */
public class RelativeDataSetLossCalculator implements ScoreCalculator<MultiLayerNetwork> {

	private static final long serialVersionUID = -7477328462428638309L;
	private DataSetIterator dataSetIterator;
    @JsonProperty
    private boolean average;
    
    public RelativeDataSetLossCalculator(){
    	
    }

    public RelativeDataSetLossCalculator(DataSetIterator dataSetIterator, boolean average) {
        this.dataSetIterator = dataSetIterator;
        this.average = average;
    }

    @Override
    public double calculateScore(MultiLayerNetwork network) {
        dataSetIterator.reset();

        double losSum = 0.0;
        int exCount = 0;
        while (dataSetIterator.hasNext()) {
            DataSet dataSet = dataSetIterator.next();
            if (dataSet == null) {
                break;
            }
            long nEx = dataSet.getFeatures().size(0);

            INDArray output = network.output(dataSet.getFeatures(), false);
            INDArray labels = dataSet.getLabels();

            INDArray score = Transforms.abs(output.sub(labels));
            score = score.div(labels);

            exCount += nEx;
            losSum += score.sumNumber().doubleValue();
        }

        if (average) {
            return losSum / exCount;
        }
        return losSum;
    }

    @Override
    public boolean minimizeScore() {
        return false;
    }
}