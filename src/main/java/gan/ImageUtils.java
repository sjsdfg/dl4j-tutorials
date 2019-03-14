package gan;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

/**
 * @author hezf
 * @date 19/1/17
 */
public class ImageUtils {

    public static void save(String path, INDArray data) {
        File file = new File(path);
        File dir = file.getParentFile();
        if (!dir.exists()) {
            dir.mkdirs();
        }
        BufferedImage bufferedImage = toBufferedImage(data);//loader.asBufferedImage(picCopy.getRow(0));
        try {
            ImageIO.write(bufferedImage, "jpg", file);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static INDArray load(String path) {
        INDArray indArray0 = null;
        try {
            indArray0 = toINDArrayBGR(ImageIO.read(new File(path)));
        } catch (IOException e) {
            e.printStackTrace();
        }

        return indArray0;
    }

    public static INDArray toINDArrayBGR(BufferedImage image) {
        int height = image.getHeight();
        int width = image.getWidth();
        int bands = image.getRaster().getNumBands();

        int[] pixels = new int[width * height];
        pixels = getRGB(image, 0, 0, width, height, pixels);
        int[] shape = new int[] {bands, height, width};

        INDArray ret2 = Nd4j.create(1, width * height * bands);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;
                int color = pixels[idx];
                int[] argb = trimRGBColor(color);
                ret2.putScalar(idx, (argb[1]) & 0xFF);
                if (bands > 1) {
                    ret2.putScalar(idx + pixels.length, (argb[2]) & 0xFF);
                }
                if (bands > 2) {
                    ret2.putScalar(idx + pixels.length * 2, (argb[3]) & 0xFF);
                }
            }
        }
        return Nd4j.expandDims(ret2.reshape(shape), 0);
    }

    public static int[] trimRGBColor(int color) {
        int[] rgb = new int[4];
        rgb[0] = (color >>24) & 0xff;
        rgb[1] = (color >> 16) & 0xff;
        rgb[2] = (color >> 8) & 0xff;
        rgb[3] = color & 0x000000ff;

        return rgb;
    }

    public static BufferedImage toBufferedImage(INDArray data) {
        long[] shape = data.shape();
        int width = (int) shape[3];
        int heith = (int) shape[2];
        int chl = (int) shape[1];

        BufferedImage image = new BufferedImage(width, heith, BufferedImage.TYPE_3BYTE_BGR);

        int[] dataPixels = data.permute(0, 3, 2, 1).getRow(0).data().asInt();

        int[] pixels = new int[width * heith];

        for (int y = 0; y < heith; y++) {
            for (int x = 0; x < width; x++) {
                int[] rgb = new int[4];
                rgb[0] = 0xff;
                rgb[1] = dataPixels[y * width + x];
                if (chl > 1) {
                    rgb[2] = dataPixels[y * width + x + pixels.length];
                }
                if (chl > 2) {
                    rgb[3] = dataPixels[y * width + x + pixels.length * 2];
                }
                pixels[y * width + x] = rgb[0] << 24 | rgb[1] << 16 | rgb[2] << 8 | rgb[3];
            }
        }
        setRGB(image, 0, 0, width, heith, pixels);
        return image;
    }

    public static int[] getRGB( BufferedImage image, int x, int y, int width, int height, int[] pixels ) {
        int type = image.getType();
        if ( type == BufferedImage.TYPE_INT_ARGB || type == BufferedImage.TYPE_INT_RGB )
            return (int [])image.getRaster().getDataElements( x, y, width, height, pixels );
        return image.getRGB( x, y, width, height, pixels, 0, width );
    }

    public static void setRGB( BufferedImage image, int x, int y, int width, int height, int[] pixels ) {
        int type = image.getType();
        if ( type == BufferedImage.TYPE_INT_ARGB || type == BufferedImage.TYPE_INT_RGB )
            image.getRaster().setDataElements( x, y, width, height, pixels );
        else
            image.setRGB( x, y, width, height, pixels, 0, width );
    }
}
