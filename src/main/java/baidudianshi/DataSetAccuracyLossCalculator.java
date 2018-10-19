package baidudianshi;

import org.deeplearning4j.earlystopping.scorecalc.ScoreCalculator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Created by Joe on 2018/1/17
 */
public class DataSetAccuracyLossCalculator implements ScoreCalculator<MultiLayerNetwork> {

	private static final long serialVersionUID = -7477328462428638309L;
	private DataSetIterator[] dataSetIterators;


    public DataSetAccuracyLossCalculator(DataSetIterator... dataSetIterators) {
        this.dataSetIterators = dataSetIterators;
    }

    @Override
    public double calculateScore(MultiLayerNetwork network) {

        double sum = 0;
        for (DataSetIterator dataSetIterator : dataSetIterators) {
            Evaluation eval = network.evaluate(dataSetIterator);
            sum += eval.accuracy();
        }

        return sum / dataSetIterators.length;
    }


    @Override
    public boolean minimizeScore() {
        // //All classification metrics should be maximized: ACCURACY, F1, PRECISION, RECALL, GMEASURE, MCC
        return false;
    }
}