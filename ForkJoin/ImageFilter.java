/**
 * Iterative nine-point image convolution filter working on linearized image.
 * In each of the NRSTEPS iteration steps, the average RGB-value of each pixel
 * in the source array is computed taking into account the pixel and its 8 neighbor
 * pixels (in 2D) and written to the destination array.
 */
public class ImageFilter {
	private int[] src;
	private int[] dst;
	private int width;
	private int height;

	private final int NRSTEPS = 100;

	public ImageFilter(int[] src, int[] dst, int w, int h) {
		this.src = src;
		this.dst = dst;
		width = w;
		height = h;
	}

	public void apply() {
		int index, pixel;
		for (int steps = 0; steps < NRSTEPS; steps++) {
			for (int i = 1; i < height - 1; i++) {
				for (int j = 1; j < width - 1; j++) {
					float rt = 0, gt = 0, bt = 0;
					for (int k = i - 1; k <= i + 1; k++) {
						index = k * width + j - 1;
						pixel = src[index];
						rt += (float) ((pixel & 0x00ff0000) >> 16);
						gt += (float) ((pixel & 0x0000ff00) >> 8);
						bt += (float) ((pixel & 0x000000ff));

						index = k * width + j;
						pixel = src[index];
						rt += (float) ((pixel & 0x00ff0000) >> 16);
						gt += (float) ((pixel & 0x0000ff00) >> 8);
						bt += (float) ((pixel & 0x000000ff));

						index = k * width + j + 1;
						pixel = src[index];
						rt += (float) ((pixel & 0x00ff0000) >> 16);
						gt += (float) ((pixel & 0x0000ff00) >> 8);
						bt += (float) ((pixel & 0x000000ff));
					}
					// Re-assemble destination pixel.
					index = i * width + j;
					int dpixel = (0xff000000) | (((int) rt / 9) << 16) | (((int) gt / 9) << 8) | (((int) bt / 9));
					dst[index] = dpixel;
				}
			}
			// swap references
			int[] help; help = src; src = dst; dst = help;
		}
	}

}