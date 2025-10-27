import streamlit as st
import os
import json
import pandas as pd
import google.generativeai as genai
from PIL import Image
import fitz  # PyMuPDF
from dotenv import load_dotenv
import tempfile
import io
import zipfile
from datetime import datetime


# Load environment variables
load_dotenv()


# Page configuration
st.set_page_config(
    page_title="Circuit Diagram Data Extractor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    .upload-text {
        text-align: center;
        padding: 2rem;
        border: 2px dashed #ccc;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)


# Initialize session state
if 'extraction_done' not in st.session_state:
    st.session_state.extraction_done = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'uploaded_filename' not in st.session_state:
    st.session_state.uploaded_filename = None
if 'validated_image' not in st.session_state:
    st.session_state.validated_image = None


# Initialize API
@st.cache_resource
def initialize_api():
    """Initialize Google Generative AI API"""
    try:
        API_KEY = os.getenv('GOOGLE_API_KEY')
        if not API_KEY:
            st.error("⚠️ GOOGLE_API_KEY not found in .env file")
            st.info("Please create a .env file with: GOOGLE_API_KEY=your_api_key_here")
            return None
        genai.configure(api_key=API_KEY)
        st.success("✅ API initialized successfully")
        return True
    except Exception as e:
        st.error(f"❌ Error initializing API: {e}")
        return None


def validate_file_type(uploaded_file):
    """Validate if uploaded file is of supported type"""
    try:
        allowed_types = ['image/png', 'image/jpg', 'image/jpeg', 'application/pdf']
        allowed_extensions = ['.png', '.jpg', '.jpeg', '.pdf']
        
        # Check MIME type
        if uploaded_file.type not in allowed_types:
            return False, f"❌ Unsupported file type: {uploaded_file.type}"
        
        # Check file extension
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension not in allowed_extensions:
            return False, f"❌ Invalid file extension: {file_extension}"
        
        # Check file size (max 10MB)
        max_size = 10 * 1024 * 1024  # 10MB in bytes
        if uploaded_file.size > max_size:
            return False, f"❌ File size exceeds 10MB limit. Current size: {uploaded_file.size / (1024 * 1024):.2f}MB"
        
        # Check if file size is too small (likely empty or corrupted)
        if uploaded_file.size < 1024:  # Less than 1KB
            return False, "❌ File appears to be empty or corrupted"
        
        return True, "✅ File type validated successfully"
    
    except Exception as e:
        return False, f"❌ File validation error: {str(e)}"


def validate_image_content(img):
    """Validate if image is readable and has proper dimensions"""
    try:
        # Check if image can be opened
        if img is None:
            return False, "❌ Unable to open image file"
        
        # Check image dimensions
        width, height = img.size
        
        if width < 100 or height < 100:
            return False, f"❌ Image dimensions too small ({width}x{height}). Minimum required: 100x100 pixels"
        
        if width > 10000 or height > 10000:
            return False, f"❌ Image dimensions too large ({width}x{height}). Maximum allowed: 10000x10000 pixels"
        
        # Check image mode
        if img.mode not in ['RGB', 'RGBA', 'L']:
            return False, f"❌ Unsupported image mode: {img.mode}"
        
        return True, "✅ Image content validated successfully"
    
    except Exception as e:
        return False, f"❌ Image validation error: {str(e)}"


def validate_electrical_diagram(img):
    """Use Gemini AI to validate if the image is an electrical circuit diagram"""
    try:
        validation_prompt = """
        Analyze this image and determine if it is an electrical or electronic circuit diagram.
        
        An electrical circuit diagram should contain:
        - Electrical/electronic component symbols (resistors, capacitors, transistors, etc.)
        - Wiring or connection lines
        - Terminal blocks or connection points
        - Labels or annotations related to electrical components
        - Circuit schematics or wiring diagrams
        
        Respond with a JSON object in this exact format:
        {
            "is_electrical_diagram": true or false,
            "confidence": "high/medium/low",
            "reason": "Brief explanation of why this is or isn't an electrical diagram"
        }
        
        If the image contains any of the following, mark it as NOT an electrical diagram:
        - Random photos, selfies, or non-technical images
        - Text documents without circuit symbols
        - Blank or mostly empty pages
        - Non-electrical diagrams (flowcharts, organizational charts, etc.)
        """
        
        validation_schema = {
            "type": "object",
            "properties": {
                "is_electrical_diagram": {"type": "boolean"},
                "confidence": {"type": "string"},
                "reason": {"type": "string"}
            },
            "required": ["is_electrical_diagram", "confidence", "reason"]
        }
        
        model = genai.GenerativeModel(
            model_name='gemini-2.0-flash-exp',
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": validation_schema
            }
        )
        
        with st.spinner('🔍 Validating if image is an electrical circuit diagram...'):
            response = model.generate_content([img, validation_prompt])
            validation_result = json.loads(response.text.strip())
        
        return validation_result
    
    except Exception as e:
        st.warning(f"⚠️ AI validation unavailable: {str(e)}")
        # Return a default response if validation fails
        return {
            "is_electrical_diagram": True,  # Allow processing to continue
            "confidence": "unknown",
            "reason": "Validation service temporarily unavailable"
        }


def process_pdf_to_image(pdf_file):
    """Convert PDF to image with enhanced error handling"""
    try:
        with st.spinner('📄 Converting PDF to image...'):
            # Validate PDF file
            if pdf_file.size == 0:
                return None, False, "❌ PDF file is empty"
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_file.read())
                tmp_path = tmp_file.name
            
            # Try to open and validate PDF
            try:
                pdf_document = fitz.open(tmp_path)
            except Exception as pdf_error:
                os.remove(tmp_path)
                return None, False, f"❌ Invalid or corrupted PDF file: {str(pdf_error)}"
            
            # Check if PDF has pages
            if pdf_document.page_count == 0:
                pdf_document.close()
                os.remove(tmp_path)
                return None, False, "❌ PDF file contains no pages"
            
            # Convert first page to image
            page = pdf_document[0]
            pix = page.get_pixmap(dpi=300)
            img_path = tmp_path.replace('.pdf', '.png')
            pix.save(img_path)
            pdf_document.close()
            
            # Clean up PDF
            os.remove(tmp_path)
            st.success('✅ PDF converted successfully')
            return img_path, True, "✅ Conversion successful"
    
    except Exception as e:
        return None, False, f"❌ Error converting PDF: {str(e)}"


def extract_outputs(img, user_prompt):
    """Extract data from circuit diagram with enhanced error handling"""
    
    system_prompt = (
        "You are a specialist in electrical and control engineering schematics. "
        "Your primary task is to analyze the provided circuit diagram image and extract ALL data from BOTH: "
        "1. The **Digital Output (DO) block** labeled 'OUTPUT', 'BINARY OUTPUT', 'WIRING: BINARY OUTPUT', 'GC (G1) OUTPUT', or 'IG-NTC-BB CONTROLLER OUTPUT' "
        "2. The **Digital Input (DI) block** labeled 'INPUT', 'BINARY INPUT', 'WIRING: BINARY INPUT', 'GC (G1) INPUT', or 'IG-NTC-BB CONTROLLER INPUT' "
        
        "For EVERY row in BOTH blocks (including rows marked 'NOT USED'), you MUST extract the following details: "
        
        "**For OUTPUT Block:** "
        "1. 'output_terminal_id' (The output terminal name, e.g., BO1, BO2, BO3... BO12) "
        "2. 'output_terminal_number' (The 1-2 digit number that appears directly next to/inside the BO box itself) "
        "3. 'output_wire_number' (Trace the horizontal line from the BO terminal - the wire number on this line) "
        "4. 'relay_name' (Extract ONLY the SHORT 2-4 LETTER ABBREVIATION CODE from the horizontal line) "
        "5. 'output_contact_number' (The 2-digit number on the LEFT side of the relay name box) "
        
        "**For INPUT Block:** "
        "1. 'input_terminal_id' (The input terminal name, e.g., BI1, BI2, BI3... BI12) "
        "2. 'input_terminal_number' (The terminal position number) "
        "3. 'input_wire_number' (The wire number on the line) "
        "4. 'tb_name' (Terminal block names like TBDR-203, TBDR-06, ECTB) "
        "5. 'input_contact_number' (The contact identifier) "
        
        "Return results as a JSON object with 'outputs' and 'inputs' arrays. Use 'N/A' for missing values."
    )


    schema = {
        "type": "object",
        "properties": {
            "outputs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "IG1000_OUTPUT": {"type": "string"},
                        "terminal": {"type": "string"},
                        "wire_number": {"type": "string"},
                        "circuit_tag": {"type": "string"},
                        "contact_number_a": {"type": "string"}
                    },
                    "required": ["IG1000_OUTPUT", "terminal", "wire_number", "circuit_tag", "contact_number_a"]
                }
            },
            "inputs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "IG1000_INPUT": {"type": "string"},
                        "terminal": {"type": "string"},
                        "wire_number": {"type": "string"},
                        "tb_name": {"type": "string"},
                        "contact_number_a": {"type": "string"}
                    },
                    "required": ["IG1000_INPUT", "terminal", "wire_number", "tb_name", "contact_number_a"]
                }
            }
        },
        "required": ["outputs", "inputs"]
    }


    try:
        model = genai.GenerativeModel(
            model_name='gemini-2.0-flash-exp',
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": schema
            },
            system_instruction=system_prompt
        )
        
        # Progress bar for extraction
        progress_bar = st.progress(0, text="🔍 Initializing AI model...")
        progress_bar.progress(25, text="📤 Sending image to Gemini API...")
        
        response = model.generate_content([img, user_prompt])
        progress_bar.progress(75, text="🧠 AI is analyzing the circuit...")
        
        json_string = response.text.strip()
        extracted_data = json.loads(json_string)
        
        progress_bar.progress(100, text="✅ Extraction completed!")
        progress_bar.empty()
        
        # Validate extracted data
        if not extracted_data.get('outputs') and not extracted_data.get('inputs'):
            st.warning("⚠️ No terminal data found in the diagram. Please ensure the image contains OUTPUT and INPUT blocks.")
        
        return extracted_data, True, None
    
    except json.JSONDecodeError as json_error:
        return {"outputs": [], "inputs": []}, False, f"❌ Failed to parse AI response: {str(json_error)}"
    
    except Exception as e:
        error_msg = str(e)
        
        # Handle specific API errors
        if "404" in error_msg or "NOT_FOUND" in error_msg:
            return {"outputs": [], "inputs": []}, False, "❌ AI model not found. Please check your API configuration."
        elif "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            return {"outputs": [], "inputs": []}, False, "❌ API quota exceeded. Please try again later."
        elif "403" in error_msg or "PERMISSION_DENIED" in error_msg:
            return {"outputs": [], "inputs": []}, False, "❌ API permission denied. Please check your API key."
        else:
            return {"outputs": [], "inputs": []}, False, f"❌ Extraction error: {error_msg}"


@st.cache_data
def convert_df_to_csv(df):
    """Convert DataFrame to CSV for download"""
    return df.to_csv(index=False).encode('utf-8')


def create_zip_file(outputs_df, inputs_df, filename):
    """Create a ZIP file containing both CSV files"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add outputs CSV
        if not outputs_df.empty:
            outputs_csv = outputs_df.to_csv(index=False)
            zip_file.writestr(f"{filename}_outputs.csv", outputs_csv)
        
        # Add inputs CSV
        if not inputs_df.empty:
            inputs_csv = inputs_df.to_csv(index=False)
            zip_file.writestr(f"{filename}_inputs.csv", inputs_csv)
    
    zip_buffer.seek(0)
    return zip_buffer


# Main App
def main():
    # Header
    st.title("⚡ Circuit Diagram Data Extractor")
    st.markdown("### Extract terminal data from electrical wiring diagrams")
    st.markdown("---")
    
    # Initialize API
    api_status = initialize_api()
    if not api_status:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        **How to use:**
        1. Upload a circuit diagram (PNG, JPG, or PDF)
        2. System validates file type and content
        3. Preview the uploaded image
        4. Click '🚀 Extract Data' button
        5. View results in interactive tables
        6. Download individual CSV or ZIP file
        
        **Supported formats:**
        - 🖼️ PNG images
        - 🖼️ JPG/JPEG images
        - 📄 PDF documents (first page)
        
        **File Requirements:**
        - Maximum size: 10MB
        - Minimum dimensions: 100x100 pixels
        - Must be an electrical circuit diagram
        """)
        
        st.divider()
        
        st.markdown("**📊 Device Information**")
        st.info("**Device Type:** IG1000")
        st.info("**Extracts:** Binary Output & Input terminals")
        
        st.divider()
        
        st.markdown("**💡 Tips**")
        st.markdown("""
        - Use high-quality images for better results
        - Ensure diagram is clearly visible
        - PDF files are auto-converted to images
        - Only upload electrical circuit diagrams
        """)
        
        st.divider()
        
        # Statistics
        if st.session_state.extraction_done and st.session_state.results:
            st.markdown("**📈 Current Session Stats**")
            outputs_count = len(st.session_state.results.get('outputs', []))
            inputs_count = len(st.session_state.results.get('inputs', []))
            st.metric("Outputs Found", outputs_count)
            st.metric("Inputs Found", inputs_count)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("📤 Upload Section")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a circuit diagram file",
            type=['png', 'jpg', 'jpeg', 'pdf'],
            help="Upload a circuit diagram image or PDF file",
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            # Step 1: Validate file type
            is_valid_type, type_message = validate_file_type(uploaded_file)
            
            if not is_valid_type:
                st.error(type_message)
                st.error("❌ **Please upload a valid file:**")
                st.markdown("""
                - PNG, JPG, JPEG images
                - PDF documents
                - File size < 10MB
                """)
                st.session_state.validated_image = None
                st.stop()
            
            # Display file info
            st.success("✅ File uploaded successfully!")
            st.info(f"**Filename:** {uploaded_file.name}")
            st.info(f"**File size:** {uploaded_file.size / 1024:.2f} KB")
            st.info(f"**Type:** {uploaded_file.type}")
            
            # Store filename
            st.session_state.uploaded_filename = uploaded_file.name.split('.')[0]
    
    with col1:
        st.subheader("🖼️ Preview Section")
        
        if uploaded_file is not None:
            # Process file
            img_path = None
            is_temp = False
            img = None
            
            try:
                if uploaded_file.type == "application/pdf":
                    st.warning("📄 PDF detected - converting to image...")
                    img_path, is_temp, conversion_msg = process_pdf_to_image(uploaded_file)
                    
                    if not is_temp:
                        st.error(conversion_msg)
                        st.error("❌ **PDF Conversion Failed**")
                        st.markdown("**Please ensure:**")
                        st.markdown("- PDF file is not corrupted")
                        st.markdown("- PDF contains at least one page")
                        st.markdown("- PDF file is a valid format")
                        st.session_state.validated_image = None
                        st.stop()
                    
                    img = Image.open(img_path)
                else:
                    try:
                        img = Image.open(uploaded_file)
                    except Exception as img_error:
                        st.error(f"❌ Failed to open image: {str(img_error)}")
                        st.error("❌ **Image Loading Failed**")
                        st.markdown("**Possible reasons:**")
                        st.markdown("- File is corrupted")
                        st.markdown("- Unsupported image format")
                        st.markdown("- File is not a valid image")
                        st.session_state.validated_image = None
                        st.stop()
                
                # Step 2: Validate image content
                is_valid_content, content_message = validate_image_content(img)
                
                if not is_valid_content:
                    st.error(content_message)
                    st.error("❌ **Image Validation Failed**")
                    if is_temp and img_path and os.path.exists(img_path):
                        os.remove(img_path)
                    st.session_state.validated_image = None
                    st.stop()
                
                # Display image with zoom capability
                st.image(img, caption="📷 Uploaded Circuit Diagram", use_container_width=True)
                
                # Step 3: Validate if it's an electrical diagram using AI
                st.markdown("---")
                validation_result = validate_electrical_diagram(img)
                
                if not validation_result['is_electrical_diagram']:
                    st.error("❌ **This does not appear to be an electrical circuit diagram!**")
                    st.error(f"**Reason:** {validation_result['reason']}")
                    st.warning("⚠️ **Please upload an image containing:**")
                    st.markdown("""
                    - Electrical component symbols
                    - Wiring or connection lines
                    - Terminal blocks
                    - Circuit schematics
                    """)
                    if is_temp and img_path and os.path.exists(img_path):
                        os.remove(img_path)
                    st.session_state.validated_image = None
                    st.stop()
                else:
                    st.success(f"✅ **Electrical diagram detected** (Confidence: {validation_result['confidence']})")
                    st.info(f"**Analysis:** {validation_result['reason']}")
                    st.session_state.validated_image = img
                
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
                st.session_state.validated_image = None
                st.stop()
        else:
            st.info("👈 Please upload a circuit diagram to begin extraction")
            st.session_state.validated_image = None
    
    # Extraction button
    st.markdown("---")
    
    if uploaded_file is not None and st.session_state.validated_image is not None:
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        
        with col_btn2:
            if st.button("🚀 Extract Data", type="primary", use_container_width=True):
                extraction_prompt = (
                    "Extract all details for the digital output and input terminal blocks "
                    "and return the results as a JSON object with proper field mappings."
                )
                
                # Run extraction
                with st.spinner('⏳ Processing your request...'):
                    results, success, error_msg = extract_outputs(st.session_state.validated_image, extraction_prompt)
                    
                    if not success:
                        st.error(error_msg)
                        st.error("❌ **Data Extraction Failed**")
                        st.markdown("**Possible solutions:**")
                        st.markdown("- Check your internet connection")
                        st.markdown("- Verify API key is valid")
                        st.markdown("- Try uploading a clearer image")
                        st.markdown("- Wait a few minutes and try again")
                    else:
                        st.session_state.results = results
                        st.session_state.extraction_done = True
                
                # Clean up temporary files
                if is_temp and img_path and os.path.exists(img_path):
                    os.remove(img_path)
    
    # Display results if extraction is done
    if st.session_state.extraction_done and st.session_state.results:
        st.markdown("---")
        st.markdown("## 📊 Extraction Results")
        
        # Create DataFrames
        outputs_df = pd.DataFrame(st.session_state.results.get('outputs', []))
        inputs_df = pd.DataFrame(st.session_state.results.get('inputs', []))
        
        # Fix duplicate column names
        if not outputs_df.empty:
            outputs_df.columns = ['IG1000 OUTPUT', 'TERMINAL', 'WIRE NO', 'RELAY NAME', 'CONTACT TERMINAL']
        if not inputs_df.empty:
            inputs_df.columns = ['IG1000 INPUT', 'TERMINAL', 'WIRE NO', 'TB NAME', 'CONTACT TERMINAL']
        
        # Success message
        st.success("✅ Data extraction completed successfully!")
        
        # Summary metrics
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("📤 Output Terminals", len(outputs_df) if not outputs_df.empty else 0)
        with col_m2:
            st.metric("📥 Input Terminals", len(inputs_df) if not inputs_df.empty else 0)
        with col_m3:
            total_terminals = (len(outputs_df) if not outputs_df.empty else 0) + (len(inputs_df) if not inputs_df.empty else 0)
            st.metric("📊 Total Terminals", total_terminals)
        
        # Tabs for outputs and inputs
        tab1, tab2, tab3 = st.tabs(["📤 Binary Outputs", "📥 Binary Inputs", "📦 Download All"])
        
        with tab1:
            st.subheader("DEVICE NAME: IG1000 - WIRING: BINARY OUTPUT")
            if not outputs_df.empty:
                st.dataframe(
                    outputs_df, 
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )
                
                # Download button
                csv_outputs = convert_df_to_csv(outputs_df)
                st.download_button(
                    label="⬇️ Download Outputs as CSV",
                    data=csv_outputs,
                    file_name=f"{st.session_state.uploaded_filename}_outputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    type="primary"
                )
            else:
                st.warning("⚠️ No output terminals found in the diagram")
        
        with tab2:
            st.subheader("DEVICE NAME: IG1000 - WIRING: BINARY INPUT")
            if not inputs_df.empty:
                st.dataframe(
                    inputs_df, 
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )
                
                # Download button
                csv_inputs = convert_df_to_csv(inputs_df)
                st.download_button(
                    label="⬇️ Download Inputs as CSV",
                    data=csv_inputs,
                    file_name=f"{st.session_state.uploaded_filename}_inputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    type="primary"
                )
            else:
                st.warning("⚠️ No input terminals found in the diagram")
        
        with tab3:
            st.subheader("📦 Download Both Files as ZIP")
            st.markdown("Download both output and input CSV files in a single ZIP archive.")
            
            if not outputs_df.empty or not inputs_df.empty:
                # Create ZIP file
                zip_data = create_zip_file(outputs_df, inputs_df, st.session_state.uploaded_filename)
                
                st.download_button(
                    label="📦 Download ZIP File (Outputs + Inputs)",
                    data=zip_data,
                    file_name=f"{st.session_state.uploaded_filename}_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    use_container_width=True,
                    type="primary"
                )
                
                st.success("✅ ZIP file ready for download!")
                st.info("💡 The ZIP file contains both outputs and inputs CSV files")
            else:
                st.warning("⚠️ No data available to create ZIP file")
        
        # Reset button
        st.markdown("---")
        col_reset1, col_reset2, col_reset3 = st.columns([1, 2, 1])
        with col_reset2:
            if st.button("🔄 Process Another Diagram", use_container_width=True):
                st.session_state.extraction_done = False
                st.session_state.results = None
                st.session_state.uploaded_filename = None
                st.session_state.validated_image = None
                st.rerun()


if __name__ == "__main__":
    main()
