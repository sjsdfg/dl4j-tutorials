package lesson1;

import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by Joe on 2017/12/27.
 * Getting and Setting Individual Values
 * 获取或者设定指定的数值
 */
public class Nd4jGetAndSet {
    public static void main(String[] args) {
        INDArray nd = Nd4j.create(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, new int[]{2, 6});
        System.out.println("打印原有数组");
        System.out.println(nd);

        /*
            获取指定索引的值
         */
        System.out.println("获取数组下标为0, 3的值");
        double value = nd.getDouble(0, 3);
        System.out.println(value);

        /*
            修改指定索引的值
         */
        System.out.println("修改数组下标为0, 3的值");
        nd.putScalar(0, 3, 100);

        /*
            使用索引迭代器遍历ndarray，使用c order
         */
        System.out.println("使用索引迭代器遍历ndarray");
        NdIndexIterator iter = new NdIndexIterator(2, 6);
        while (iter.hasNext()) {
            int[] nextIndex = iter.next();
            double nextVal = nd.getDouble(nextIndex);

            System.out.println(nextVal);
        }
    }
}
