import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

public class ParallelFJImageFilter {
	private int[] src;
	private int[] dst;
	private final int width;
	private final int height;

	private final int STEPS_NUMBER = 100;

	public ParallelFJImageFilter(int[] src, int[] dst, int width, int height) {
		this.src = src;
		this.dst = dst;
		this.width = width;
		this.height = height;
	}

	public void apply(int threads) {
		ForkJoinPool forkJoinPool = new ForkJoinPool(threads);
		for (int i = 0; i < STEPS_NUMBER; i++) {
			ParallelFJImageFilterAction action = new ParallelFJImageFilterAction(src, dst, 0, src.length, width);
			forkJoinPool.invoke(action);
			while(!forkJoinPool.isQuiescent()) {}
			int[] help; help = src; src = dst; dst = help;
		}
	}

	private static class ParallelFJImageFilterAction extends RecursiveAction {

		private static final int THRESHOLD = 100000;

		private final int[] src;
		private final int[] dst;
		private final int start;
		private final int length;
		private final int width;

		public ParallelFJImageFilterAction(int[] src, int[] dst, int start, int length, int width) {
			this.src = src;
			this.dst = dst;
			this.start = start;
			this.length = length;
			this.width = width;
		}

		@Override
		protected void compute() {
			if (length <= THRESHOLD || getSurplusQueuedTaskCount() > 3) {
				process();
			} else {
				int halfLength = length / 2;
				new ParallelFJImageFilterAction(src, dst, start, halfLength, width).fork();
				new ParallelFJImageFilterAction(src, dst, start + halfLength, length - halfLength, width).fork();
			}
		}

		private void process() {
			for (int i = start; i < start + length; i++) {
				int rowIndex = i % width;
				if (i < width || i >= src.length - width || rowIndex == 0 || rowIndex == width - 1) {
					continue;
				}
				float[] rgb = new float[] {0, 0, 0};
				addPixel(src[i - width], rgb);
				addPixel(src[i + width], rgb);
				addPixel(src[i + 1], rgb);
				addPixel(src[i - 1], rgb);
				addPixel(src[i - width - 1], rgb);
				addPixel(src[i - width + 1], rgb);
				addPixel(src[i + width - 1], rgb);
				addPixel(src[i + width + 1], rgb);
				addPixel(src[i], rgb);

				int dstPixel = (0xff000000) | (((int) rgb[0] / 9) << 16) | (((int) rgb[1] / 9) << 8) | (((int) rgb[2] / 9));
				dst[i] = dstPixel;
			}
		}

		private void addPixel(int pixel, float[] rgb) {
			rgb[0] += (float) ((pixel & 0x00ff0000) >> 16);
			rgb[1] += (float) ((pixel & 0x0000ff00) >> 8);
			rgb[2] += (float) ((pixel & 0x000000ff));
		}
	}
}

