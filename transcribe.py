import os
import google.generativeai as genai
import fitz # PyMuPDF
from PIL import Image
import io

# Set up your Gemini API key from environment variables for security
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

def convert_pdf_to_images(pdf_path):
    """
    Converts a PDF file into a list of PIL Image objects.
    """
    images = []
    try:
        pdf_document = fitz.open(pdf_path)
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        return images
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return []

def get_gemini_model():
    """
    Returns a Gemini model that supports image and text input.
    """
    return genai.GenerativeModel('gemini-1.5-flash')

def get_text_from_file(file_path):
    """
    Transcribes text from a PDF or PNG file using the Gemini API.
    """
    model = get_gemini_model()
    
    extracted_text = ""
    file_extension = file_path.split('.')[-1].lower()
    
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"
    
    print(f"Processing file: {file_path}")
    
    if file_extension in ['png', 'jpg', 'jpeg']:
        try:
            image = Image.open(file_path)
            processed_image = image.convert('L') # Convert to grayscale
            processed_image = processed_image.point(lambda x: 0 if x < 128 else 255, '1') # Apply binary thresholding
            prompt = [
                "You are an arabic transcriptionist. Transcribe the Arabic text from this image. Keep all diacritics (harakat). Keep it exactly how the input it, dont change, dont add, dont do anything different from the input, just word for word, letter by letter, harakat by harakat, transcribing it, after you are done, re map it all together, and once done with that do it 3 more times. You are an expert Arabic transcriptionist. Your task is to transcribe the Arabic text from this image with 100% accuracy. The output must ONLY contain the transcribed text. Do not add any commentary, explanations, or information not directly visible in the image. Preserve all diacritics (harakat) and the exact spelling. The text is written from right to left.",
                image
            ]
            response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.0))
            extracted_text = response.text
        except Exception as e:
            extracted_text = f"Error processing PNG file: {e}"
            print(extracted_text)
            
    elif file_extension == 'pdf':
        images = convert_pdf_to_images(file_path)
        if not images:
            extracted_text = "Error: Could not process PDF file."
        else:
            print(f"PDF converted to {len(images)} pages.")
            for i, img in enumerate(images):
                print(f"Transcribing page {i+1}...")
                try:
                    prompt = [
                        "Transcribe the Arabic text from this image. Keep all diacritics (harakat).",
                        img
                    ]
                    response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.0))
                    extracted_text += f"\n--- Page {i+1} ---\n" + response.text
                except Exception as e:
                    print(f"Error transcribing page {i+1}: {e}")
                    extracted_text += f"\n--- Error on Page {i+1} ---\n"
    else:
        extracted_text = "Error: Unsupported file type. Please use .pdf or .png."

    return extracted_text

def main():
    """
    Main function to run the script.
    """
    # List of input files you want to process
    input_files = ["input1.png", "input2.png", "input3.png", "input4.png", "input5.png", "input6.png", "input7.png"]  # Add your filenames here
    output_file = "output.txt"
    
    # Open the output file in write mode to start fresh
    with open(output_file, "w", encoding="utf-8") as f:
        print("Starting transcription process...")
        for input_file in input_files:
            # Get the transcribed text for each file
            transcribed_content = get_text_from_file(input_file)
            
            # Write a header for the current file and then the content
            f.write(f"\n\n--- Transcription for {input_file} ---\n")
            f.write(transcribed_content)
            
    print(f"\nTranscription complete for all files. The combined result is saved in '{output_file}'")

if __name__ == "__main__":
    main()