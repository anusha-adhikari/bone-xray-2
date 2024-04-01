import streamlit as st
from PIL import Image, ImageEnhance
import cv2
import numpy as np
import base64
from streamlit_image_zoom import image_zoom  
from keras.models import load_model
from torch import nn
import torch.nn.functional as F
import torch

@st.cache(allow_output_mutation=True)
def load_req_model(m):
    if m == 1:
        model_url = "vgg19_canny1.h5"
    if m == 2:
        model_url = "vgg19_morph1.h5"
        
    model = load_model(model_url)
    return model

class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, bias=False)

        Gx = torch.tensor([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]])
        Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return x
    
def np_img_to_tensor(img):
    if len(img.shape) == 2:
        img = img[..., np.newaxis]
    img_tensor = torch.from_numpy(img)
    img_tensor = img_tensor.permute(2, 0, 1)
    img_tensor = torch.unsqueeze(img_tensor, 0)
    return img_tensor

def tensor_to_np_img(img_tensor):
    img = img_tensor.cpu().permute(0, 2, 3, 1).numpy()
    return img[0, ...]  

def sobel_torch_version(img_np, torch_sobel):
    img_tensor = np_img_to_tensor(np.float32(img_np))
    
    
    padding = (1, 1, 1, 1)  
    img_tensor = F.pad(img_tensor, padding)
    
    img_edged = tensor_to_np_img(torch_sobel(img_tensor))
    img_edged = np.squeeze(img_edged)
    return img_edged

def sob(rgb_orig, stype, draw_bbox = False, bounding_box = ((0,0), (0, 0))):
    rgb_orig = np.array(rgb_orig)
    if len(rgb_orig.shape) > 2:
            rgb_orig = cv2.cvtColor(rgb_orig, cv2.COLOR_BGR2GRAY)
    torch_sobel = Sobel()
    
    rgb_edged = sobel_torch_version(rgb_orig, torch_sobel=torch_sobel)
    
    rgb_edged_cv2_x = cv2.Sobel(rgb_orig, cv2.CV_64F, 1, 0, ksize=3)
    rgb_edged_cv2_y = cv2.Sobel(rgb_orig, cv2.CV_64F, 0, 1, ksize=3)
    
    rgb_edged_cv2 = np.sqrt(np.square(rgb_edged_cv2_x), np.square(rgb_edged_cv2_y))
    if stype == "Type 1":
        mod = rgb_edged_cv2 / np.max(rgb_edged_cv2)
    if stype == "Type 2":
        mod = rgb_edged / np.max(rgb_edged)
    if stype == "Desired Type":
        st.write("     Please choose a valid option - :blue[Type 1] or :blue[Type 2]")
        st.stop()
    mod = (mod * 255).astype(np.uint8)
    if draw_bbox:
        mod = draw_bounding_box(mod, bounding_box)

    return mod

def morphological_processing_with_canny(image, threshold1, threshold2):
    umat_image = cv2.UMat(image)
    kernel = np.ones((2, 2), np.uint8)
    canny_edges = cv2.Canny(umat_image, threshold1=threshold1, threshold2=threshold2)
    canny_edges = np.asarray(canny_edges.get())
    gradient = cv2.morphologyEx(canny_edges, cv2.MORPH_GRADIENT, kernel)
    return gradient

def morphological_processing(image):
    kernel = np.ones((2, 2), np.uint8)
    gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    return gradient

def adjust_contrast(image, factor):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def adjust_brightness_contrast(image, alpha, beta):
    return cv2.addWeighted(image, alpha, image, 0, beta)

def modify_image(image, t1, t2, option, draw_bbox = False, bounding_box = ((0,0), (0, 0))):
    if option == "Morphological":
        enhanced_contrast = adjust_contrast(image, 1.5)
        enhanced_contrast = np.array(enhanced_contrast)

        x = morphological_processing(enhanced_contrast)
        kernel = np.ones((3, 3), np.uint8)
        gradient = cv2.morphologyEx(x, cv2.MORPH_GRADIENT, kernel)

        t1 = 1 + (t1 * 0.01)
        enhanced_image = adjust_brightness_contrast(gradient, t1, t2)

        enhanced_image = np.array(enhanced_image)
        if len(enhanced_image.shape) > 2:  
            enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
        
        if draw_bbox:
            enhanced_image = draw_bounding_box(enhanced_image, bounding_box)

        return enhanced_image

    if option == "Canny Edge":
        image = np.array(image)
        modified_image = morphological_processing_with_canny(image, t1, t2)
        if draw_bbox:
            modified_image = draw_bounding_box(modified_image, bounding_box)

        return modified_image
    
def draw_bounding_box(image, bounding_box):
    drawn_image = image.copy()
    cv2.rectangle(drawn_image, tuple(bounding_box[0]), tuple(bounding_box[1]), (255, 255, 0), 2)
    return drawn_image


def download_image(image):
    pil_image = Image.fromarray(image)
    temp_file_path = "modified_image.png"
    pil_image.save(temp_file_path, "PNG")

    try:
        with open(temp_file_path, "rb") as file:
            contents = file.read()

        encoded_file = base64.b64encode(contents).decode()
        href = f'<a href="data:file/png;base64,{encoded_file}" download="modified_image.png">Download modified image</a>'
        st.markdown(href, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred during download: {e}")

def main():
    st.set_page_config(layout='wide')

    left_co, cent_co,last_co = st.columns(3)
    with cent_co:
        st.image("logo-1.png", width=200)

    st.title("Image :blue[Uploader], :blue[Modifier], and :blue[Downloader]")

    st.caption('1. Upload an X-Ray image of bones in the sidebar.')
    st.caption('2. Choose preferred type of edge detection')
    st.caption('3. Canny Edge detection : Adjust the two threshold values to obtain a clear outline.')
    st.caption('4. Morphological Edge detection : Adjust the contrast and brightness values to obtain a clear outline.')
    st.caption('5. Sobel Edge detection : No adjustment required to obtain a clear outline.')
    st.caption('6. Zoom factor : Adjust the value so that hovering over the modified image, zooms that part of the image (1.00 - no zoom; 5.00 - max zoom).')
    st.caption('7. Bounding box can be made to appear by clicking the checkbox.')
    st.caption('8. Adjust the X1 & X2 -> horizontal adjustment, and Y1 & Y2 -> vertical adjustments in the placement of the bounding box on the desired location of the modified image.')
    st.caption('9. Processing the bounding box gives the extracted region from inside the bounding box and classifies if the area in the bounding box has a fracture or not!')
    st.caption('10. Modified image can be downloaded with/ without bounding box depending upon need.')

    col1, col2 = st.columns(2)

    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "JPG", "JPEG"])

    option = st.sidebar.selectbox(
        "Choose:",
        ("Edge Detection type", "Sobel", "Morphological", "Canny Edge"),
        index = 0
    )

    if option is not None and option == "Edge Detection type":
        st.write("Please choose a valid option - :blue[Yes] or :blue[No]")
        st.stop()

    st.sidebar.subheader("Parameters")
    if option == "Morphological":
        threshold = st.sidebar.slider("Contrast increase by : ", min_value=0, max_value=100, value=40)
        threshold2 = st.sidebar.slider("Brightness increase by adding : ", min_value=0, max_value=100, value=5)

    if option == "Canny Edge":
        threshold = st.sidebar.slider("Threshold 1 : ", min_value=0, max_value=200, value=12)
        threshold2 = st.sidebar.slider("Threshold 2 : ", min_value=0, max_value=400, value=76)

    if option == "Sobel":
        stype = st.sidebar.selectbox(
        "Choose:",
        ("Desired Type", "Type 1", "Type 2"),
        index=0
    )

    zoom_factor = st.sidebar.slider("Zoom Factor : ", min_value=1.0, max_value=5.0, value=2.0)

    
    draw_bbox = st.sidebar.checkbox("Draw bounding box on the image to point out where you feel pain", value=False)
    if draw_bbox == True:
        st.sidebar.subheader("Bounding Box Parameters")
        st.sidebar.caption("Horizontal:")
        x1 = st.sidebar.slider("X1", min_value=0, max_value=2000, value=50)
        x2 = st.sidebar.slider("X2", min_value=0, max_value=2000, value=150)
        st.sidebar.caption("Vertical:")
        y1 = st.sidebar.slider("Y1", min_value=0, max_value=2000, value=50)
        y2 = st.sidebar.slider("Y2", min_value=0, max_value=2000, value=150)

        bounding_box = ((x1, y1), (x2, y2))

    with col1:
        st.subheader(":blue[Uploaded Image]")
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

        with col2:
            st.subheader(":blue[Modified Image]")
            if uploaded_file is not None:
                
                if draw_bbox == True:
                    if option != "Sobel":
                        modified_image = modify_image(image, threshold, threshold2, option, draw_bbox, bounding_box)
                    else:
                        modified_image = sob(image, stype, draw_bbox, bounding_box)
                else:
                    if option != "Sobel":
                        modified_image = modify_image(image, threshold, threshold2, option)
                    else:
                        modified_image = sob(image, stype)
                        
                
                image_zoom(modified_image, mode="mousemove", size=512, zoom_factor=zoom_factor)

                
                        
                if draw_bbox == True:
                    if option == "Canny Edge" or option == "Morphological":
                        if st.sidebar.button("Process Bounding Box to find out if you have a fracture"):
                            
                            min_y = min(y1, y2)
                            max_y = max(y1, y2)
                            min_x = min(x1, x2)
                            max_x = max(x1, x2)
                            region_of_interest = modified_image[min_y:max_y, min_x:max_x]
    
                                
                            st.sidebar.image(region_of_interest, caption="Extracted Region",
                                                    use_column_width=True)
                            if option == "Canny Edge":
    
                                model = load_req_model(1)
    
                                img = cv2.cvtColor(region_of_interest, cv2.COLOR_GRAY2RGB)
                                img = cv2.resize(img, (100, 100))
                                img = img/255
                                img = np.reshape(img, ((1,)+img.shape))
    
                                if model.predict(img) >= 0.5:
                                    st.error('Fracture detected')
                                else:
                                    st.success('No fracture detected')
    
                            if option == "Morphological":
    
                                model = load_req_model(2)
    
                                img = cv2.cvtColor(region_of_interest, cv2.COLOR_GRAY2RGB)
                                img = cv2.resize(img, (100, 100))
                                img = img/255
                                img = np.reshape(img, ((1,)+img.shape))
    
                                if model.predict(img) >= 0.5:
                                    st.error('Fracture detected')
                                else:
                                    st.success('No fracture detected')

                    if option == "Sobel":
                            
                        if st.sidebar.button("Process Bounding Box to see Extracted Region"):
                            
                            min_y = min(y1, y2)
                            max_y = max(y1, y2)
                            min_x = min(x1, x2)
                            max_x = max(x1, x2)
                            region_of_interest = modified_image[min_y:max_y, min_x:max_x]
        
                            st.sidebar.image(region_of_interest, caption="Extracted Region",
                                                        use_column_width=True)
                                
                            st.sidebar.image(modified_image, caption="Modified Image", use_column_width=True)

                download_button = st.button("Download Modified Image")
                if download_button:
                    download_image(modified_image)

if __name__ == "__main__":
    main()
