package lesson7;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.util.LinkedList;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * 回归用的迭代器
 * Created by Joe on 2017/7/20.
 */
public class RegIterator implements DataSetIterator {

    //每批次的训练数据组数
    private int miniBatch;
    //每组训练数据长度（DailyData的个数）
    private int exampleLength;
    //需要拟合的向量个数
    private int vectorSize;
    //数据集
    private List<Data> dataList;
    //存放剩余数据的index信息
    private List<Integer> dataRecord;

    private DataSetPreProcessor preProcessor;

    /**
     * 迭代器的构造函数, 数据集无标签用这个构造函数

     * @param miniBatch          微批次
     * @param exampleLength      训练的数据长度
     */
    public RegIterator(List<Data> dataList, int vectorSize, int miniBatch, int exampleLength) {
        //参数的初始化
        this.miniBatch = miniBatch;
        this.exampleLength = exampleLength;
        this.vectorSize = vectorSize;


        //对dataList进行初始化，便于后面读取数据后对数据的存入
        this.dataList = dataList;

        //对dataRecord进行初始化，便于后面数据的填入
        this.dataRecord = new LinkedList<>();
        resetDataRecord();
    }



    private void resetDataRecord() {
        dataRecord.clear();
        int total = dataList.size() / exampleLength;
        //如果dataList.size()不能被exampleLength整除，则需要多+1
        if (dataList.size() % exampleLength != 0 ) {
            total += 1;
        }

        for (int i = 0; i < total; i++) {
            dataRecord.add(i * exampleLength);
        }
    }

    @Override
    public DataSet next(int num) {
        if (dataRecord.size() < 0) {
            throw new NoSuchElementException("List元素已经用完");
        }

        int actualBatchSize = Math.min(num, dataRecord.size());
        int actualLength = Math.min(exampleLength, dataList.size()-dataRecord.get(0));

        INDArray input = Nd4j.create(new int[]{actualBatchSize, vectorSize, actualLength}, 'f');
        INDArray label = Nd4j.create(new int[]{actualBatchSize, 1, actualLength}, 'f');


        //获取每批次的训练数据和训练标签
        for (int i = 0; i < actualBatchSize; i++) {
            int index = dataRecord.remove(0);
            int endIndex = Math.min(index + actualLength, dataList.size()-1);

            for (int j = index; j < endIndex; j++) {
                //获取数据信息
                Data curData = dataList.get(j);
                //构造训练向量
                int c = (j + actualLength) - endIndex;
                double[] features = curData.getDatas();
                for (int dimension = 0; dimension < vectorSize; dimension++) {
                    input.putScalar(new int[]{i, dimension, c}, features[dimension]);
                }

                //构造label向量
                label.putScalar(new int[]{i, 0, c}, features[features.length - 1]);
            }

            if(dataRecord.size() <= 0) {
                break;
            }
        }

        DataSet temp = new DataSet(input, label);
        if (preProcessor != null) {
            preProcessor.preProcess(temp);
        }
        return temp;
    }

    @Override
    public int totalExamples() {
        return 0;
    }

    @Override
    public int inputColumns() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public int totalOutcomes() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public boolean resetSupported() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public boolean asyncSupported() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public void reset() {
        resetDataRecord();
    }

    @Override
    public int batch() {
        return miniBatch;
    }

    @Override
    public int cursor() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public int numExamples() {
        return 0;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return this.preProcessor;
    }

    @Override
    public List<String> getLabels() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public boolean hasNext() {
        return dataRecord.size() > 0;
    }

    @Override
    public DataSet next() {
        return next(miniBatch);
    }
}
