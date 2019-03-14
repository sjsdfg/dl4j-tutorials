package gan;

import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;


/**
 * @author hezf
 * @date 19/1/18
 */
public class InOutputPlatPreProcessor implements InputPreProcessor {

    public static boolean gen = false;

    private int width;
    private int height;
    private int channel;
    public InOutputPlatPreProcessor(int width, int height, int channel) {
        this.width = width;
        this.height = height;
        this.channel = channel;
    }

    private static double[] toDbArr(INDArray data) {
        long[] shape = data.shape();
        int width = (int) shape[3];
        int heith = (int) shape[2];

        double[] dataDb = data.permute(0, 3, 2, 1).getRow(0).data().asDouble();
        return dataDb;
    }

    public static double[] temp2Arr;

    @Override
    public INDArray preProcess(INDArray input, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        if (gen) {
            INDArray picCopy = input.dup();
            picCopy = picCopy.reshape(new int[] {1, channel, height, width});
            temp2Arr = toDbArr(picCopy);
            ImageUtils.save("/myself/tmp/dl4j/gan/data/train/0/0.jpg", picCopy.mul(255));
            System.out.println("生成图片已经保存");
        }
        return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, input);
    }


    @Override
    public INDArray backprop(INDArray output, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        return workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, output);
    }

    @Override
    public InputPreProcessor clone() {
        try {
            InOutputPlatPreProcessor clone = (InOutputPlatPreProcessor) super.clone();
            return clone;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public InputType getOutputType(InputType inputType) {
        return inputType;
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState, int minibatchSize) {
        return new Pair<>(maskArray, currentMaskState);
    }
}
