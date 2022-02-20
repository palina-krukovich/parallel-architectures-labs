import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;

public class TestImageFilter {

	private static final int[] THREADS = new int[] {1, 2, 4, 8, 16, 32};
	private static final double INITIAL_SPEED_UP = 0.7;

	private static int[] sample;
	private static long sampleTime;

	public static void main(String[] args) throws Exception {
		setOut(args[1]);

		String srcFileName = args[0];
		BufferedImage img = IOImageTool.loadImage(srcFileName);
		System.out.printf("Source image: %s\n", srcFileName);

		int w = img.getWidth();
		int h = img.getHeight();
		System.out.printf("Image size is %dx%d\n\n", w, h);

		runSequentialFilter(img, w, h, String.format("Filtered%s", srcFileName));

		System.out.printf("Available processors: %d\n\n", Runtime.getRuntime().availableProcessors());

		for (int i = 0; i < THREADS.length; i++) {
			runParallelFilter(img, w, h, String.format("ParallelFiltered%d_%s", THREADS[i], srcFileName), THREADS[i]);
		}
	}

	private static void runSequentialFilter(BufferedImage image, int width, int height, String outFileName) throws IOException {
		int[] src = IOImageTool.getRGB(image, width, height);
		int[] dst = new int[src.length];

		System.out.println("Starting sequential image filter.");
		long startTime = System.currentTimeMillis();
		ImageFilter filter = new ImageFilter(src, dst, width, height);
		filter.apply();
		sample = dst;
		long endTime = System.currentTimeMillis();

		sampleTime = endTime - startTime;
		System.out.printf("Sequential image filter took %d milliseconds.\n", sampleTime);

		IOImageTool.saveImage(dst, width, height, outFileName);
		System.out.printf("Output image: %s\n\n", outFileName);
	}

	private static void runParallelFilter(BufferedImage image, int width, int height, String outFileName, int threadsNumber) throws IOException {
		int[] src = IOImageTool.getRGB(image, width, height);
		int[] dst = new int[src.length];

		System.out.printf("Starting parallel image filter using %d threads.\n", threadsNumber);
		long startTime = System.currentTimeMillis();
		ParallelFJImageFilter filter = new ParallelFJImageFilter(src, dst, width, height);
		filter.apply(threadsNumber);
		long endTime = System.currentTimeMillis();

		long t = endTime - startTime;
		double speedUp = sampleTime / (double) t;
		double sampleSpeedUp = INITIAL_SPEED_UP * threadsNumber;
		boolean ok = speedUp >= sampleSpeedUp;

		System.out.printf("Parallel image filter took %d milliseconds using %d threads.\n", t, threadsNumber);
		System.out.println(verifyResult(sample, dst) ? "Output image verified successfully!" : "INCORRECT IMAGE!");
		System.out.printf("Speedup: %.5f %s (%s %.1f)\n", speedUp, ok ? "ok" : "not ok", ok ? ">=" : "<", sampleSpeedUp);

		IOImageTool.saveImage(dst, width, height, outFileName);

		System.out.printf("Output image: %s\n\n", outFileName);
	}

	private static boolean verifyResult(int[] sample, int[] result) {
		return Arrays.equals(sample, result);
	}

	private static void setOut(String outFileName) throws FileNotFoundException {
		PrintStream out = new PrintStream(outFileName);
		System.setOut(out);
	}

	private static class IOImageTool {
		public static BufferedImage loadImage(String fileName) {
			try {
				File file = new File(fileName);
				return ImageIO.read(file);
			}
			catch (ArrayIndexOutOfBoundsException e) {
				System.out.println("Usage: java TestAll <image-file>");
			}
			catch (IOException e) {
				System.out.printf("Error reading image file %s !%n\n", fileName);
			}
			System.exit(1);
			return null;
		}

		public static void saveImage(int[] arr, int width, int height, String fileName) throws IOException {
			BufferedImage dstImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
			setRGB(dstImage, width, height, arr);

			File file = new File(fileName);
			ImageIO.write(dstImage, "jpg", file);
		}

		public static int[] getRGB(BufferedImage image, int width, int height) {
			return image.getRGB(0, 0, width, height, null, 0, width);
		}

		public static void setRGB(BufferedImage image, int width, int height, int[] arr) {
			image.setRGB(0, 0, width, height, arr, 0, width);
		}
	}
}
