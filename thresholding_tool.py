import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.signal import argrelextrema
import time

class ThresholdingApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.auto_mode = True
        self.threshold = 127
        self.last_auto_threshold = 127
        
        # Visualization Config
        self.frame_w = 320
        self.frame_h = 240
        self.plot_w = 640
        self.plot_h = 300
        self.window_name = "CV Thresholding Tool - Integrated"
        
        # Matplotlib Setup (Non-interactive backend)
        self.fig, self.ax = plt.subplots(figsize=(self.plot_w/100, self.plot_h/100), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        
        self.x_axis = np.arange(256)
        self.line_raw, = self.ax.plot(self.x_axis, np.zeros(256), color='gray', alpha=0.5, label='Raw')
        self.line_smooth, = self.ax.plot(self.x_axis, np.zeros(256), color='blue', linewidth=2, label='Smoothed')
        self.threshold_line = self.ax.axvline(x=127, color='red', linestyle='--', linewidth=2, label='Thresh')
        self.minima_plot, = self.ax.plot([], [], 'ro', markersize=8)
        self.peaks_plot, = self.ax.plot([], [], 'go', markersize=6)
        
        self.ax.set_xlim(0, 255)
        self.ax.set_ylim(0, 1000)
        self.ax.set_title("Intensity Histogram (Click to set Manual Threshold)")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper right', fontsize='small')
        self.fig.tight_layout(pad=1.0)

        # UI Layout state
        self.button_rect = (260, 10, 120, 30) # x, y, w, h (relative to top row center mostly)
    
    def compute_histogram(self, gray):
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        return hist.flatten()

    def smooth_histogram(self, hist, kernel_size=15):
        kernel = np.ones(kernel_size) / kernel_size
        return np.convolve(hist, kernel, mode='same')

    def find_local_minima(self, smoothed_hist, last_threshold=None):
        minima_indices = argrelextrema(smoothed_hist, np.less, order=10)[0]
        valid_minima = [m for m in minima_indices if 20 < m < 235]
        
        if not valid_minima:
            return minima_indices, 127
            
        if last_threshold is None:
            primary_threshold = min(valid_minima, key=lambda x: smoothed_hist[x])
        else:
            closest = min(valid_minima, key=lambda x: abs(x - last_threshold))
            deepest = min(valid_minima, key=lambda x: smoothed_hist[x])
            
            curr_depth = smoothed_hist[closest]
            best_depth = smoothed_hist[deepest]
            
            if best_depth < (curr_depth * 0.80) and abs(deepest - last_threshold) > 20:
                primary_threshold = deepest
            else:
                primary_threshold = closest
                
        return minima_indices, int(primary_threshold)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check Button Click (Top area logic depends on layout)
            # Layout: 
            # [Original 320x240] [Binary 320x240]
            # [       Plot 640x300              ]
            
            # Button is drawn on the combined frame. 
            # Let's say button is centered at top for visibility.
            bx, by, bw, bh = self.button_rect
            if bx <= x <= bx + bw and by <= y <= by + bh:
                self.auto_mode = not self.auto_mode
                return

            # Check Graph Click
            # Graph starts at y = 240
            if y > self.frame_h:
                # Map x to threshold
                # Plot width is 640, Range is 0-255.
                # Need to account for plot margins (axes).
                # A simple approximation mapping:
                # The axis usually occupies e.g. 10% to 90% of the figure width.
                # To be precise, we use the axes transform.
                
                # However, since we can't easily invert the mpl transform from here efficiently without some math,
                # let's use a linear approximation assuming tight_layout/margins.
                # Or better: Just map the width 0-640 -> 0-255 loosely, or accept that clicks near edges are inaccurate.
                
                # Better approach:
                # The axis bbox in pixels can be retrieved, but it changes.
                # Let's try simple linear mapping first.
                # Graph area approx 12% padding left, 10% right?
                # Let's assume the data area is roughly x=50 to x=600.
                
                # We can enforce specific margins in mpl setup to make this robust.
                # But for now:
                graph_x = x
                if graph_x < 50: graph_x = 50
                if graph_x > 590: graph_x = 590
                
                # Map 50-590 to 0-255
                val = int((graph_x - 50) / (590 - 50) * 255)
                self.threshold = max(0, min(255, val))
                self.auto_mode = False # Switch to manual on interaction

    def run(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        last_time = time.time()
        frame_count = 0
        fps = 0
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret: break
                
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Processing
                hist = self.compute_histogram(gray)
                hist_smooth = self.smooth_histogram(hist)
                minima, auto_t = self.find_local_minima(hist_smooth, self.last_auto_threshold)
                self.last_auto_threshold = auto_t
                
                if self.auto_mode:
                    self.threshold = auto_t
                    
                binary = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)[1]
                
                # --- Visualization Construction ---
                
                # 1. Update Plot
                self.line_raw.set_ydata(hist)
                self.line_smooth.set_ydata(hist_smooth)
                self.threshold_line.set_xdata([self.threshold])
                
                if len(minima) > 0:
                    self.minima_plot.set_data(minima, hist_smooth[minima])
                else:
                    self.minima_plot.set_data([], [])
                    
                peak_idxs = argrelextrema(hist_smooth, np.greater, order=10)[0]
                if len(peak_idxs) > 0:
                    self.peaks_plot.set_data(peak_idxs, hist_smooth[peak_idxs])
                else:
                    self.peaks_plot.set_data([], [])
                
                self.ax.set_ylim(0, max(np.max(hist_smooth)*1.1, 100))
                
                # Render Plot to Buffer
                self.canvas.draw()
                buf = self.canvas.buffer_rgba()
                plot_img_rgba = np.asarray(buf)
                plot_img = cv2.cvtColor(plot_img_rgba, cv2.COLOR_RGBA2BGR)
                
                # 2. Resize Frames
                frame_small = cv2.resize(frame, (self.frame_w, self.frame_h))
                binary_small = cv2.resize(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), (self.frame_w, self.frame_h))
                
                # 3. Overlays
                cv2.putText(frame_small, "Original", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(binary_small, "Binary", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # 4. Collage Assembly
                top_row = np.hstack([frame_small, binary_small])
                collage = np.vstack([top_row, plot_img])
                
                # 5. Draw UI Elements on Collage
                # Button - Center Top
                bx, by, bw, bh = self.button_rect
                btn_color = (0, 255, 0) if self.auto_mode else (0, 165, 255) # Green vs Orange
                cv2.rectangle(collage, (bx, by), (bx+bw, by+bh), btn_color, -1)
                cv2.putText(collage, "AUTO" if self.auto_mode else "MANUAL", (bx+15, by+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # Info Text
                cv2.putText(collage, f"Threshold: {self.threshold}", (10, self.frame_h - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                cv2.putText(collage, f"FPS: {fps}", (self.plot_w - 80, self.frame_h - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

                cv2.imshow(self.window_name, collage)
                
                # FPS Counter
                frame_count += 1
                if time.time() - last_time >= 1.0:
                    fps = frame_count
                    frame_count = 0
                    last_time = time.time()
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                elif key == ord('m'):
                    self.auto_mode = not self.auto_mode
                    
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            plt.close(self.fig)
            print("App Closed")

if __name__ == "__main__":
    app = ThresholdingApp()
    app.run()
