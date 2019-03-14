package gan;

import java.util.Arrays;

import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;

/**
 * 用于GAN的选取真实/噪声图片的InputPreProcessor
 * 
 * @author liweigu
 *
 */
public class GanCnnInputPreProcessor implements InputPreProcessor {
	private static final long serialVersionUID = 6362224879456600896L;
	private long inputHeight;
	private long inputWidth;
	private long numChannels;
	private long[] shape;
	// 在preProcess里设置isFirstInputEmpty，在backprop里使用isFirstInputEmpty
	private boolean isRInputEmpty;
	private static boolean printLog = false; // false, true

	public static boolean save = false;

	/**
	 * 默认构造函数，支持序列化。
	 */
	public GanCnnInputPreProcessor() {
	}

	/**
	 * 将 [(channels x 2) x rows x columns] 转换为 [channels x rows x columns]
	 *
	 * @param inputHeight the columns
	 * @param inputWidth the rows
	 * @param numChannels the channels
	 */
	public GanCnnInputPreProcessor(long inputHeight, long inputWidth,
			long numChannels) {
		this.inputHeight = inputHeight;
		this.inputWidth = inputWidth;
		this.numChannels = numChannels;
	}

	@Override
	public INDArray preProcess(INDArray input, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
		// [1 , numChannels * 2, inputHeight, inputWidth]
		this.shape = input.shape();
		// System.out.println("input = " + input);
		// System.out.println("input.sumNumber() = " + input.sumNumber());
		if (printLog) {
			System.out.println("this.shape = " + Arrays.toString(this.shape));
		}
		// Input: 4d activations (CNN)
		// Output: 4d activations (CNN)
		if (input.rank() != 4) {
			throw new IllegalArgumentException(
					"Invalid input: expect CNN activations with rank 4 (received input with shape " + Arrays.toString(input.shape()) + ")");
		}

		if (input.ordering() != 'c' || !Shape.hasDefaultStridesForShape(input)) {
			input = input.dup('c');
			// input = workspaceMgr.dup(ArrayType.ACTIVATIONS, input, 'c');
		}

		// 将2张CNN转为1张CNN
		INDArray newInput = Nd4j.zeros(shape[0], shape[1] / 2, shape[2], shape[3]);
		for (int i = 0; i < shape[0]; i++) {
			// [numChannels * 2, inputHeight, inputWidth]: z + r
			INDArray multyImage = input.get(NDArrayIndex.point(i), NDArrayIndex.all());
			// System.out.println("multyImage.sumNumber() = " + multyImage.sumNumber());
			// [numChannels * 1, inputHeight, inputWidth]
			INDArray newMultyImage = newInput.getRow(i);

			int newRowIndex = 0;
			for (int j = 0; j < shape[1] / 2; j++) {
				// [inputHeight, inputWidth]
				INDArray rImageWH = null;
				if (j == 0) {
					// 第一步，读取rImageWH，并判断它是否为空
					// "z-input", "r-input"
					rImageWH = multyImage.get(NDArrayIndex.point(j + shape[1] / 2), NDArrayIndex.all());
					// System.out.println("rImageWH.sumNumber() = " + rImageWH.sumNumber());
					double firstPixelValue = rImageWH.getDouble(0, 0);
					if (firstPixelValue != -9999) {
						this.isRInputEmpty = false;
					} else {
						this.isRInputEmpty = true;
					}
				}

				if (!this.isRInputEmpty) {
					if (rImageWH == null) {
						rImageWH = multyImage.get(NDArrayIndex.point(j + shape[1] / 2), NDArrayIndex.all());
					}
					// System.out.println("newRowIndex = " + newRowIndex);
					newMultyImage.putRow(newRowIndex, rImageWH);
					// System.out.println("newMultyImage.sumNumber() = " + newMultyImage.sumNumber());
				} else {
					INDArray zImageWH = multyImage.get(NDArrayIndex.point(j), NDArrayIndex.all());
					newMultyImage.putRow(newRowIndex, zImageWH);
				}
				newRowIndex++;
			}

			newInput.putRow(i, newMultyImage);
		}
		// System.out.println("newInput = " + newInput);
		// System.out.println("newInput.sumNumber() = " + newInput.sumNumber());

		// return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, newInput);
		if (save) {
			ImageUtils.save("/myself/tmp/dl4j/gan/data/train/0/0.jpg", newInput.dup().mul(255));
		}
		return newInput;
	}

	/**
	 * Reverse the preProcess during backprop. Process Gradient/epsilons before passing them to the layer below.
	 * 
	 * @param epsilons which is a pair of the gradient and epsilon. 后一层的epsilon是目标函数对前一层的激活函数值的偏导。要是为0的话 会导致前一层的权重的梯度为0。
	 * @param miniBatchSize
	 * @param workspaceMgr
	 * @return the reverse of the pre preProcess step (if any). Note that the returned array should be placed in
	 *         {@link ArrayType#ACTIVATION_GRAD} workspace via the workspace manager
	 */
	@Override
	public INDArray backprop(INDArray epsilons, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
		if (printLog) {
			System.out.println("this.isRInputEmpty = " + this.isRInputEmpty);
		}
		// System.out.println("epsilons.sumNumber() = " + epsilons.sumNumber());
		// System.out.println("epsilons.shape() = " + Arrays.toString(epsilons.shape()));
		if (epsilons.ordering() != 'c' || !Shape.hasDefaultStridesForShape(epsilons)) {
			epsilons = workspaceMgr.dup(ArrayType.ACTIVATION_GRAD, epsilons, 'c');
		}

		if (shape == null || ArrayUtil.prod(shape) != epsilons.length()) {
			INDArray newEpsilons = Nd4j.zeros(shape[0], shape[1], shape[2], shape[3]);
			for (int i = 0; i < shape[0]; i++) {
				// [numChannels * 1, inputHeight, inputWidth]
				INDArray multyImage = epsilons.getRow(i);
				// [numChannels * 2, inputHeight, inputWidth]
				INDArray newMultyImage = newEpsilons.getRow(i);

				// INDArray zeroINDArray = Nd4j.zeros(shape[3], shape[2]);

				// System.out.println("shape[1] = " + shape[1]);
				for (int j = 0; j < shape[1] / 2; j++) {
					// [inputHeight, inputWidth]
					INDArray imageWH = multyImage.getRow(j);
					// System.out.println("imageWH = " + imageWH);
					if (printLog) {
						System.out.println("imageWH.shape() = " + Arrays.toString(imageWH.shape()));
					}
					if (this.isRInputEmpty) {
						newMultyImage.putRow(j, imageWH);
						// 其他的默认是0值
						// newMultyImage.putRow(j + shape[1] / 2, zeroINDArray);
					} else {
						// newMultyImage.putRow(j, zeroINDArray);
						// 其他的默认是0值
						newMultyImage.putRow(j + shape[1] / 2, imageWH);
					}
				}
			}
			if (printLog) {
				System.out.println("newEpsilons.shape() = " + Arrays.toString(newEpsilons.shape()));
				System.out.println("newEpsilons = " + newEpsilons);
			}

			return newEpsilons;
			// return newEpsilons.reshape('c', newEpsilons.size(0), numChannels * 2, inputHeight, inputWidth);
		}

		if (printLog) {
			System.out.println("reshape to " + shape[1] * 2);
		}
		// return workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, epsilons.reshape('c', shape[0], shape[1] * 2, shape[2], shape[3]));
		return epsilons;
		// return epsilons.reshape('c', shape[0], shape[1] * 2, shape[2], shape[3]);
	}

	@Override
	public GanCnnInputPreProcessor clone() {
		try {
			GanCnnInputPreProcessor clone = (GanCnnInputPreProcessor) super.clone();
			if (clone.shape != null)
				clone.shape = clone.shape.clone();
			return clone;
		} catch (CloneNotSupportedException e) {
			throw new RuntimeException(e);
		}
	}

	@Override
	public InputType getOutputType(InputType inputType) {
		switch (inputType.getType()) {
		case CNN:
			InputType.InputTypeConvolutional c2 = (InputType.InputTypeConvolutional) inputType;

			if (c2.getChannels() != numChannels || c2.getHeight() != inputHeight || c2.getWidth() != inputWidth) {
				throw new IllegalStateException("Invalid input: Got CNN input type with (d,w,h)=(" + c2.getChannels() + "," + c2.getWidth() + ","
						+ c2.getHeight() + ") but expected (" + numChannels + "," + inputHeight + "," + inputWidth + ")");
			}
			return c2;
		default:
			throw new IllegalStateException("Invalid input type: got " + inputType);
		}
	}

	@Override
	public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState, int minibatchSize) {
		// Pass-through, unmodified (assuming here that it's a 1d mask array - one value per example)
		return new Pair<>(maskArray, currentMaskState);
	}

	public void setPath(String path) {

	}
}
