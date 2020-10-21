import cv2

def regions_of_interest(fg_mask, min_size, max_size, aspect_ratio, add_context=True):
    """ Searches ROIs based on backgound subtraction

    Args:
        bgs (cv::BackgroundSubtractor)
        frame (cv2::Image): Consecutive frame
        add_context (Boolean): whether to add context (padding) or not 

    Returns:
        rects (array([x,y,w,h])): Array of found rectangles
    """

    # RETR_EXTERNAL: only extreme outer contours
    # CHAIN_APPROX_SIMPLE: compresses segments to only return 4 endpoints for each contour
    img, contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    # contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

    # both not need, but returned by cv3.4.8
    # del img
    # del hierarchy
    
    rects = []
    for cnt in contours:
        rect = cv2.boundingRect(cnt)
        if not is_roi_of_interest(rect, min_size, max_size, aspect_ratio):
            continue

        rect = contexted_crop(rect, fg_mask.shape, 10)
        
        # convert roi to xyxy format and append
        x,y,w,h = rect
        rects.append([x,y,x+w,y+h]) 

    return rects


MIN_SIZE, MAX_SIZE, ASPECT_RATIO = 15, 2000, 0.2
def is_roi_of_interest(rectangle, min_size, max_size, aspect_ratio):
    """ Blob analysis as suggested by [Aslani2013OpticalFB].

    Args:
        rect (array [x,y,w,h]): the rectangle to be analysed
        min_size (int): the minimum size of w and h
        max_size (int): the maximum size of w and h
        aspect_ratio (float): the minimum ratio of the blob

    Returns:
        True (bool), if rect matches requirements
            False (bool), elsewise
    """

    _,_,width,height = rectangle

    if min(width, height) < min_size: return False
    if max(width, height) > max_size: return False
    if width/height < aspect_ratio: return False
    if height/width < aspect_ratio: return False

    return True


def contexted_crop(rect, imageshape, padding):
    """ Adds contextextual information to the rectangle to improve reasoning as suggested by [arXiv:1612.04402 [cs]] 

    Args:
        rect ([x,y,w,h]): Found minimum rectangle
        imageshape (cv2::Image.shape): The imageshape, used to check boundaries and extract crop
        padding (Int): Amount of padding to apply to rectangle

    Returns:
        cv::Image: the contexted crop matching desired dimension
    """

    x, y, w, h = rect
    max_h, max_w = imageshape

    """
    # approach by percantage...
    pad_x = int(w/100*padding)
    pad_y = int(h/100*padding)
    
    # pad_x = int(max_w/100)
    # pad_y = int(max_h/100)
    
    x = int(max(x - pad_x, 0))
    y = int(max(y - pad_y, 0))
    w = int(min(w + pad_x*2, max_w))
    h = int(min(h + pad_y*2, max_h))
    """
    # approach by abs pixel
    x = int(max(x - padding, 0))
    y = int(max(y - padding, 0))
    w = int(min(w + padding*2, max_w))
    h = int(min(h + padding*2, max_h))
    
    
    return [x, y, w, h]
