import os
from PIL import Image
import numpy as np
import wave
import struct
from fpdf import FPDF

def create_test_image():
    # Create a larger 500x500 grayscale image
    img = Image.new('L', (500, 500), color=255)
    pixels = np.array(img)
    
    # Add simulated damage pattern
    pixels[150:350, 150:350] = 100  # Larger damage area
    pixels[200:300, 200:300] = 50   # More intense damage
    
    # Save the image
    img = Image.fromarray(pixels)
    img.save('test_files/damage_photo.jpg')
    print("Created test image: damage_photo.jpg")

def create_test_audio():
    # Create a simple WAV file with silence
    with wave.open('test_files/statement.wav', 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(44100)
        for _ in range(44100):  # 1 second of silence
            value = struct.pack('h', 0)
            f.writeframes(value)
    print("Created test audio: statement.wav")

def create_test_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Add test content
    pdf.cell(200, 10, txt="Insurance Claim Document", ln=1, align='C')
    pdf.cell(200, 10, txt="Date: 01/15/2024", ln=1, align='L')
    pdf.cell(200, 10, txt="Policy #: ABC123456", ln=1, align='L')
    pdf.cell(200, 10, txt="Claim Amount: $5,000.00", ln=1, align='L')
    
    pdf.output("test_files/claim_document.pdf")
    print("Created test PDF: claim_document.pdf")

def setup_test_files():
    # Create test_files directory if it doesn't exist
    if not os.path.exists('test_files'):
        os.makedirs('test_files')
        print("Created test_files directory")
    
    # Create test files
    create_test_image()
    create_test_audio()
    create_test_pdf()
    
    print("\nAll test files created successfully!")

if __name__ == "__main__":
    setup_test_files()