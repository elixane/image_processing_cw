import cv2
import os
import numpy as np  

dir = 'driving_images'
output_dir = 'Results'
os.makedirs(output_dir, exist_ok=True)


def denoise_sections(image):
    '''Apply non-local means denoising to an image by processing overlapping small sections.'''

    # Define height and width of image
    height, width = image.shape[:2]
    
    # Create accumulation arrays for colour images
    accumulated = np.zeros((height, width, 3), dtype=np.float32)
    weight_accum = np.zeros((height, width), dtype=np.float32)

    # Define size of each section
    section = (32, 32)
    section_w, section_h = section 

    # Define number of pixels to overlap between sections
    overlap = 16
    
    # Create a 2D weight mask using a Hanning window
    weight_y = np.hanning(section_h)
    weight_x = np.hanning(section_w)
    weight_mask = np.outer(weight_y, weight_x)
    weight_mask = weight_mask / np.max(weight_mask)  # Normalize to max of 1
    
    # Define steps (non-overlapping portion of each section)
    step_x = section_w - overlap
    step_y = section_h - overlap
    
    # Iterate over the image with overlapping sections
    for y in range(0, height, step_y):
        for x in range(0, width, step_x):
            # Determine tile boundaries.
            x_end = min(x + section_w, width)
            y_end = min(y + section_h, height)
            tile = image[y:y_end, x:x_end]
            
            # Adjust the weight mask if it is at the border
            curr_weight_mask = weight_mask[:(y_end - y), :(x_end - x)]
            
            # Apply non-local means denoising to the current section
            denoised_section = cv2.fastNlMeansDenoisingColored(tile, None, 10, 10, 7, 21)
            denoised_section = denoised_section.astype(np.float32)
            
            # Expand current weight mask to cover the 3 colour channels
            cwm_expanded = np.expand_dims(curr_weight_mask, axis=2)
            weighted_section = denoised_section * cwm_expanded
            
            # Accumulate the weighted sections and weights
            accumulated[y:y_end, x:x_end] += weighted_section
            weight_accum[y:y_end, x:x_end] += curr_weight_mask
    
    # Compute the final blended image
    result = accumulated / weight_accum[..., np.newaxis]
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result

def colour_correct(image):
    '''Correct colour channel imbalance using the Max-RGB (White Patch) method, 
    which scales each channel based on its maximum value.'''

    # Create a copy of the image and convert it to float32 for precision in calculations
    result = image.copy().astype(np.float32)

    # Compute the maximum value for each color channel (B, G, R)
    maxB = np.max(result[:, :, 0])
    maxG = np.max(result[:, :, 1])
    maxR = np.max(result[:, :, 2])

    # Determine the overall maximum value among all channels
    max_all = max(maxB, maxG, maxR)

    # Scale each channel so that its maximum value becomes the overall maximum.
    result[:, :, 0] = result[:, :, 0] * (max_all / maxB)
    result[:, :, 1] = result[:, :, 1] * (max_all / maxG)
    result[:, :, 2] = result[:, :, 2] * (max_all / maxR)

    # Clip values to ensure they remain in the valid range [0, 255]
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result

    
def gamma_correction(image):
    '''Apply gamma correction to adjust image brightness.'''

    # Set gamma values (<1 brightens the image, >1 darkens it)
    target = 127
    gamma_min = 0.25
    gamma_max = 2.0

    # Compute average brightness of image
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(grey)

    # Compute adaptive gamma
    if avg_brightness == 0:
        gamma = gamma_max
    else:
        gamma = target / avg_brightness
        # Clamp gamma to the allowed range.
        gamma = max(min(gamma, gamma_max), gamma_min)
    
    # Use gamma correction formula to correct brightness 
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
    corrected = cv2.LUT(image, table)
    
    return corrected


def rotate(image):
    '''Rotate image using edge detection'''

    # Convert to grayscale
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use adaptive thresholding for better edge detection
    thresh = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 2)

    # Find contours (external edges only)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours are found, return the original image
    if not contours:
        return image

    # Get the largest contour (outer boundary)
    contour = max(contours, key=cv2.contourArea)

    # Get the bounding box and compute angle
    rect = cv2.minAreaRect(contour)
    (x, y), (w, h), angle = rect

    # Standardize angle to always rotate < 45° clockwise
    if w < h:
        angle = -angle
    else:
        angle = 90 - angle

    if angle > 45:
        angle -= 90

    # Fix cases where image ends up sideways
    if abs(angle) > 45:  
        angle += 90 

    # Get image dimensions
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Get rotation matrix and apply affine transformation
    rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated

def warp(image, frame_size=(256, 256)):
    '''Warp image by projecting its outer edges onto a 256x256 frame'''

    # Convert to grayscale
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(grey, (5, 5), 0)

    # List of parameter sets to try in adaptiveThreshold
    threshold_params = [(17, 4), (19, 2), (21, 2), (21, 0)] 
    
    for a, b in threshold_params:
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, a, b)

        # Apply dynamic edge detection
        median_intensity = np.median(grey)
        lower_thresh = max(10, 0.4 * median_intensity)
        upper_thresh = min(255, 1.6 * median_intensity)
        edges = cv2.Canny(thresh, lower_thresh, upper_thresh)

        # Morphological closing to fill small gaps in edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue  # Retry with the next set of parameters

        # Sort contours by area and select the top 5 largest contours
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        largest_contour = None

        # Iterate through the largest contours to find a valid quadrilateral
        for contour in contours:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 4:  # We need exactly 4 corners
                largest_contour = approx
                # print(f"thresholding params {a}, {b}")
                break

        if largest_contour is not None:
            break  # Stop trying new parameters if a valid quadrilateral is found

    if largest_contour is None:
        print("Not enough outer corners detected after multiple attempts!")
        return image  # Return rotated image if no quadrilateral is found

    # Sort the four points to match (top-left, top-right, bottom-left, bottom-right)
    approx = largest_contour.reshape(4, 2)
    sorted_pts = sorted(approx, key=lambda x: (x[1], x[0]))  # Sort by y first, then x

    if sorted_pts[0][0] > sorted_pts[1][0]:  # Ensure top corners are in correct order
        sorted_pts[0], sorted_pts[1] = sorted_pts[1], sorted_pts[0]

    if sorted_pts[2][0] > sorted_pts[3][0]:  # Ensure bottom corners are in correct order
        sorted_pts[2], sorted_pts[3] = sorted_pts[3], sorted_pts[2]

    # Define source points
    src_pts = np.float32(sorted_pts)
    # Compute the center of the quadrilateral
    center = np.mean(src_pts, axis=0)
    # Scaling factor to ensure image has no borders
    scale_factor = 0.95
    src_pts_expanded = center + scale_factor * (src_pts - center)

    # Define destination points to fit the 256x256 frame
    dst_pts = np.float32([[0, 0], [frame_size[0], 0], [0, frame_size[1]], [frame_size[0], frame_size[1]]])

    # Compute the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts_expanded, dst_pts)

    # Warp the image to the fixed frame size
    warped = cv2.warpPerspective(image, M, frame_size)

    return warped


def circle(image):
    '''Detect and inpaint the largest black circle in the top-right corner using Hough Circle Transform.'''

    # Convert to grayscale
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve circle detection
    blurred = cv2.GaussianBlur(grey, (9, 9), 2)

    # Detect circles using Hough Transform
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                               param1=80, param2=30, minRadius=5, maxRadius=35)

    # Create an empty mask for inpainting
    mask = np.zeros_like(grey)

    if circles is not None:
        # Convert circle parameters to integers
        circles = np.uint16(np.around(circles))

        # Find the largest circle (by radius)
        largest_circle = max(circles[0, :], key=lambda c: c[2])  # c[2] is the radius

        x, y, r = largest_circle

        # Ensure the circle is in the top-right region
        h, w = image.shape[:2]
        if x > w * 0.5 and y < h * 0.5:
            # Draw the detected circle onto the mask
            new_r = int(r*1.2)  # Increase radius by 5 pixels
            cv2.circle(mask, (x, y), new_r, 255, thickness=cv2.FILLED)

            # Inpaint the detected circle
            inpainted = cv2.inpaint(image, mask, inpaintRadius=50, flags=cv2.INPAINT_TELEA)
            return inpainted
        
    print("No circles detected!")
    return image  # Return original if no circles are found

def shape(image):
    '''Detect and inpaint the largest connected black shape in the top-right corner.'''

    h, w = image.shape[:2]
    # Crop top-right portion of image
    crop_x = int(w*0.5)
    crop_y = 0
    crop = image[crop_y:, crop_x:]  #top-right half
    
    # Convert to grey and threshold
    grey_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, mask_crop = cv2.threshold(grey_crop, 15, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological ops to clean noise
    kernel = np.ones((3,3), np.uint8)
    mask_crop = cv2.morphologyEx(mask_crop, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Find largest connected component in mask_crop
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_crop, 8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # skip background (label=0)
        # Create a mask the same size as the original image
        mask = np.zeros((h, w), dtype=np.uint8)
        # Put the largest component in the correct place
        mask_section = (labels == largest_label).astype(np.uint8)*255
        mask[crop_y:, crop_x:] = mask_section
        
        # Expand the mask by dilating it to cover a slightly larger area
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.dilate(mask, dilation_kernel, iterations=1)

        # Inpaint
        return cv2.inpaint(image, mask, inpaintRadius=60, flags=cv2.INPAINT_TELEA)
    else:
        # Return original image if no large black region found
        return image

def inpaint_circle(image):
    # Detect largest circle through Hough method
    inpainted = circle(image)  
    if inpainted is image:
        # If no circle is detected find largest connected component instead 
        inpainted = shape(image)
    return inpainted

# Apply image processing to each image in the input folder
for image in os.listdir(dir):
    if not image.lower().endswith('.jpg'):
        continue  

    # Load the image
    img_path = os.path.join(dir, image)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Error loading image: {img_path}")
        continue

    # Rotate the image
    img = rotate(img)

    # Warp the image
    img = warp(img)

    # Reduce noise 
    img = denoise_sections(img)

    # Apply gamma correction 
    img = gamma_correction(img)

    # Apply white patch balancing
    img = colour_correct(img)

    # Apply circle inpainting 
    img = inpaint_circle(img)

    # Write the processed image to the Results folder
    output_path = os.path.join(output_dir, image)
    cv2.imwrite(output_path, img)
    print(f"Saved: {output_path}")

