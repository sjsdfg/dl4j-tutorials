package lesson7;

import java.time.LocalDateTime;
import java.util.Arrays;

/**
 * 用于存储
 * Created by Joe on 2017/9/15.
 */
public class Data {
    /**
     * 发生时间
     */
    private LocalDateTime occurTime;
    /**
     * 测点数据
     */
    private double[] datas;
    /**
     * 标记 0-正常 1-异常
     */
    private int label;

    public Data() {
    }

    public Data(LocalDateTime occurTime, double[] datas) {
        this.occurTime = occurTime;
        this.datas = datas;
    }

    public Data(LocalDateTime occurTime, double[] datas, int label) {
        this.occurTime = occurTime;
        this.datas = datas;
        this.label = label;
    }

    public LocalDateTime getOccurTime() {
        return occurTime;
    }

    public void setOccurTime(LocalDateTime occurTime) {
        this.occurTime = occurTime;
    }

    public double[] getDatas() {
        return datas;
    }

    public void setDatas(double[] datas) {
        this.datas = datas;
    }

    public int getLabel() {
        return label;
    }

    public void setLabel(int label) {
        this.label = label;
    }

    @Override
    public String toString() {
        return occurTime +
                "," + Arrays.toString(datas).replace("[", "").replace("]", "").replace(" ", "") +
                "," + label;
    }
}
