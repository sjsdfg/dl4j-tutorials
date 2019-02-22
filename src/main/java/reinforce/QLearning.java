package reinforce;

import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * QLearning 示例
 * 讲解视频 : https://www.bilibili.com/video/av16921335/?p=6
 * 原有 python 代码实现：https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/1_command_line_reinforcement_learning/treasure_on_right.py
 */
public class QLearning {
    // the length of the 1 dimensional world
    private final static int N_STATE = 6;
    // available actions 可选操作
    private final static String[] ACTIONS = new String[]{
            "left", "right"
    };
    // greedy police 贪心选择策略。 90%概率选择最优解。10%概率随机选择行为
    private final static double EPSILON = 0.9;
    // learning rate 学习率
    private final static double ALPHA = 0.1;
    // discount factor
    private final static double GAMMA = 0.9;
    // maximum episodes 最大回合数，等价 epoch
    private final static double MAX_EPISODES = 13;
    // fresh time for one move 每一步的刷新时间
    private final static double FRESH_TIME = 0.3;

    public static void main(String[] args) {
        INDArray qTable = buildQTable();
        for (int i = 0; i < MAX_EPISODES; i++) {
            int stepCounter = 0;
            int S = 0;
            boolean terminated = false;
            updateEnv(S, i, stepCounter);
            while (!terminated) {
                int A = chooseAction(S, qTable);
                int[] result = getEnvBack(S, ACTIONS[A]);
                int S_ = result[0], R = result[1];
                double qPredict = qTable.getDouble(S, A), qTarget;
                if (S_ != Integer.MAX_VALUE) {
                    qTarget = R + GAMMA * qTable.getRow(S_).maxNumber().doubleValue();
                } else {
                    qTarget = R;
                    terminated = true;
                }
                qTable.put(S, A, qPredict + ALPHA * (qTarget - qPredict));
                S = S_;
                stepCounter++;
                updateEnv(S, i, stepCounter);
            }
        }
        System.out.println(qTable);
    }

    public static INDArray buildQTable() {
        return Nd4j.zeros(N_STATE, ACTIONS.length);
    }

    public static int chooseAction(int state, INDArray qTable) {
        INDArray stateAction = qTable.getRow(state);
        if (Math.random() > EPSILON || allZeros(stateAction)) {
            return Math.random() > 0.5 ? 1 : 0;
        }
        return Nd4j.getBlasWrapper().iamax(stateAction);
    }

    private static boolean allZeros(INDArray array) {
        NdIndexIterator iter = new NdIndexIterator(array.shape());
        while (iter.hasNext()) {
            double nextVal = array.getDouble(iter.next());
            if (nextVal != 0) {
                return false;
            }
        }
        return true;
    }

    private static int[] getEnvBack(int S, String A) {
        int S_, R;
        if ("right".equalsIgnoreCase(A)) {
            if (S == N_STATE - 2) {
                S_ = Integer.MAX_VALUE;
                R = 1;
            } else {
                S_ = S + 1;
                R = 0;
            }
        } else {
            R = 0;
            if (S == 0) {
                S_ = 0;
            } else {
                S_ = S - 1;
            }
        }
        return new int[]{S_, R};
    }

    private static void updateEnv(int S, int episode, int stepCounter) {
        char[] envList = initEnv();
        if (S == Integer.MAX_VALUE) {
            System.out.println(String.format("Episode %d: total_steps = %d", episode, stepCounter));
        } else {
            envList[S] = 'o';
            System.out.println(new String(envList));
        }
    }

    private static char[] initEnv() {
        char[] envList = new char[N_STATE];
        for (int i = 0; i < N_STATE - 1; i++) {
            envList[i] = '-';
        }
        envList[N_STATE - 1] = 'T';
        return envList;
    }
}
