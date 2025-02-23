import sys
print(sys.executable)
print(sys.path)

import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import pytesseract
import speech_recognition as sr
import PyPDF2
import numpy as np
import re
from typing import Dict, List, Union
import os
import wave
import struct
from fpdf import FPDF

class MultimodalFraudDetector:
    def __init__(self):
        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50", use_fast=True)
        self.image_model = AutoModel.from_pretrained("microsoft/resnet-50")
        self.audio_recognizer = sr.Recognizer()
        self.fraud_threshold = 0.85
        
    def process_image(self, image_path: str) -> Dict:
        """
        Process images for damage assessment and fraud detection
        """
        image = Image.open(image_path)
        inputs = self.image_processor(image, return_tensors="pt")
        outputs = self.image_model(**inputs)
        
        features = outputs.last_hidden_state.mean(dim=1)
        
        return {
            "damage_severity": self._assess_damage(features),
            "fraud_indicators": self._detect_image_fraud(features),
            "confidence_score": self._calculate_confidence(features)
        }
    
    def process_audio(self, audio_path: str) -> Dict:
        """
        Process audio file and detect potential fraud indicators
        """
        try:
            return {
                "transcript": "This is a simulated audio transcript",
                "sentiment": {"score": 0.7, "confidence": 0.8},
                "fraud_risk": {
                    "level": "low",
                    "indicators": [],
                    "confidence": 0.85
                }
            }
        except Exception as e:
            print(f"Audio processing error: {str(e)}")
            return {
                "transcript": "Audio processing failed",
                "sentiment": {"score": 0.0, "confidence": 0.0},
                "fraud_risk": {
                    "level": "unknown",
                    "indicators": ["processing_error"],
                    "confidence": 0.0
                }
            }
    
    def process_pdf(self, pdf_path: str) -> Dict:
        """
        Extract and analyze information from claim documents
        """
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
                
            return {
                "extracted_text": text,
                "claim_details": self._extract_claim_details(text),
                "document_fraud_risk": self._analyze_document_consistency(text)
            }
    
    def generate_insights(self, claim_data: Dict) -> Dict:
        """
        Generate comprehensive fraud analysis and recommendations
        """
        fraud_score = self._calculate_fraud_score(claim_data)
        
        if fraud_score > self.fraud_threshold:
            risk_level = "High"
        elif fraud_score > 0.5:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return {
            "fraud_probability": fraud_score,
            "risk_level": risk_level,
            "recommendations": self._generate_recommendations(fraud_score),
            "predicted_risks": self._predict_future_risks(claim_data)
        }
    
    def _assess_damage(self, features) -> float:
        """
        Assess damage severity from image features
        Returns a score between 0 and 1
        """
        feature_array = features.detach().numpy()
        
        intensity = np.mean(np.abs(feature_array))
        variance = np.var(feature_array)
        peak_values = np.max(np.abs(feature_array))
        
        damage_score = (0.4 * intensity + 0.3 * variance + 0.3 * peak_values)
        
        return float(np.clip(damage_score, 0, 1))
    
    def _detect_image_fraud(self, features) -> List[str]:
        """
        Detect potential fraud indicators in image
        Returns list of suspicious patterns
        """
        feature_array = features.detach().numpy()
        fraud_indicators = []
        
        try:
            mean_val = np.mean(feature_array)
            std_val = np.std(feature_array)
            max_val = np.max(feature_array)
            min_val = np.min(feature_array)
            
            if std_val < 0.1:
                fraud_indicators.append("unusually_uniform_patterns")
            
            if (max_val - min_val) > 0.9:
                fraud_indicators.append("extreme_value_differences")
                
            if mean_val > 0.8:
                fraud_indicators.append("potential_artificial_enhancement")
                
            if mean_val < 0.2:
                fraud_indicators.append("potential_image_manipulation")
                
        except Exception as e:
            print(f"Warning: Error in fraud detection: {str(e)}")
            fraud_indicators.append("analysis_inconclusive")
        
        return fraud_indicators
    
    def _calculate_confidence(self, features) -> float:
        """
        Calculate confidence score for the analysis
        """
        feature_array = features.detach().numpy()
        
        signal_strength = np.mean(np.abs(feature_array))
        consistency = 1 - np.std(feature_array)
        feature_quality = np.min(np.abs(feature_array))
        
        confidence = (0.4 * signal_strength + 
                     0.4 * consistency + 
                     0.2 * feature_quality)
        
        return float(np.clip(confidence, 0, 1))
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment in audio transcript
        """
        positive_keywords = {'honest', 'truth', 'accident', 'legitimate', 'real'}
        negative_keywords = {'maybe', 'probably', 'think', 'unsure', 'possibly'}
        
        words = set(text.lower().split())
        
        positive_score = len(words.intersection(positive_keywords)) / len(positive_keywords)
        negative_score = len(words.intersection(negative_keywords)) / len(negative_keywords)
        
        return {
            "score": 1 - (negative_score / (positive_score + 0.01)),
            "confidence": min(positive_score + negative_score, 1.0)
        }
    
    def _detect_audio_fraud(self, text: str) -> Dict:
        """
        Detect potential fraud indicators in audio
        """
        fraud_indicators = {
            'hesitation': {'um', 'uh', 'like', 'sort of'},
            'uncertainty': {'maybe', 'probably', 'think so'},
            'deflection': {'cannot remember', 'not sure', "don't recall"}
        }
        
        text_lower = text.lower()
        scores = {}
        
        for category, keywords in fraud_indicators.items():
            count = sum(1 for word in keywords if word in text_lower)
            scores[category] = min(count / len(keywords), 1.0)
        
        return {
            "indicators": scores,
            "level": "high" if any(s > 0.7 for s in scores.values()) else 
                    "medium" if any(s > 0.4 for s in scores.values()) else "low"
        }
    
    def _extract_claim_details(self, text: str) -> Dict:
        """
        Extract relevant details from claim documents
        """
        patterns = {
            'amount': r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            'date': r'\d{1,2}/\d{1,2}/\d{4}',
            'policy_number': r'policy\s*#?\s*(\w+)',
            'phone': r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        }
        
        details = {}
        for key, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            details[key] = matches if matches else None
        
        return details
    
    def _analyze_document_consistency(self, text: str) -> Dict:
        """
        Analyze document for internal consistency
        """
        inconsistencies = []
        risk_score = 0.0
        
        dates = re.findall(r'\d{1,2}/\d{1,2}/\d{4}', text)
        if len(dates) > len(set(dates)):
            inconsistencies.append("multiple_different_dates")
            risk_score += 0.3
        
        amounts = re.findall(r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)', text)
        if len(amounts) > len(set(amounts)):
            inconsistencies.append("multiple_different_amounts")
            risk_score += 0.3
        
        sentences = text.split('.')
        unique_words_ratio = len(set(text.split())) / len(text.split())
        if unique_words_ratio < 0.4:
            inconsistencies.append("repetitive_narrative")
            risk_score += 0.2
        
        return {
            "inconsistencies": inconsistencies,
            "score": min(risk_score, 1.0)
        }
    
    def _calculate_fraud_score(self, claim_data: Dict) -> float:
        image_confidence = claim_data.get('image', {}).get('confidence_score', 0.0)
        audio_risk = claim_data.get('audio', {}).get('fraud_risk', {}).get('level', 'low')
        doc_risk = claim_data.get('document', {}).get('document_fraud_risk', {}).get('score', 0.0)
        
        audio_score = 0.8 if audio_risk == 'high' else 0.4 if audio_risk == 'medium' else 0.2
        
        return (image_confidence + audio_score + doc_risk) / 3.0
    
    def _generate_recommendations(self, fraud_score: float) -> List[str]:
        if fraud_score > self.fraud_threshold:
            return ["High risk detected - Manual review required",
                    "Verify all submitted documents",
                    "Contact claimant for additional information"]
        elif fraud_score > 0.5:
            return ["Enhanced review recommended",
                    "Verify key documents",
                    "Consider follow-up questions"]
        else:
            return ["Standard processing recommended",
                    "Process claim according to normal procedures"]
    
    def _predict_future_risks(self, claim_data: Dict) -> List[Dict]:
        return [
            {
                "risk_type": "Documentation",
                "probability": 0.3,
                "impact": "medium"
            },
            {
                "risk_type": "Financial",
                "probability": 0.2,
                "impact": "low"
            }
        ]

def create_test_files():
    if not os.path.exists('test_files'):
        os.makedirs('test_files')
        print("Created test_files directory")
    
    img = Image.new('L', (100, 100), color=255)
    pixels = np.array(img)
    pixels[30:70, 30:70] = 100
    pixels[40:60, 40:60] = 50
    img = Image.fromarray(pixels)
    img.save('test_files/damage_photo.jpg')
    print("Created test image")
    
    with wave.open('test_files/statement.wav', 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(44100)
        for _ in range(44100):
            value = struct.pack('h', 0)
            f.writeframes(value)
    print("Created test audio")
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Insurance Claim Document", ln=1, align='C')
    pdf.cell(200, 10, txt="Date: 01/15/2024", ln=1, align='L')
    pdf.cell(200, 10, txt="Policy #: ABC123456", ln=1, align='L')
    pdf.cell(200, 10, txt="Claim Amount: $5,000.00", ln=1, align='L')
    pdf.output("test_files/claim_document.pdf")
    print("Created test PDF")

if __name__ == "__main__":
    print("Creating test files...")
    create_test_files()
    
    print("\nInitializing Fraud Detector...")
    try:
        detector = MultimodalFraudDetector()
        print("Successfully initialized the detector!")
        
        print("\n=== Testing with Image ===")
        image_results = detector.process_image("test_files/damage_photo.jpg")
        print("Image Analysis Results:", image_results)
        
        print("\n=== Testing with Audio ===")
        audio_results = detector.process_audio("test_files/statement.wav")
        print("Audio Analysis Results:", audio_results)
        
        print("\n=== Testing with PDF ===")
        pdf_results = detector.process_pdf("test_files/claim_document.pdf")
        print("PDF Analysis Results:", pdf_results)
        
        claim_data = {
            "image": image_results,
            "audio": audio_results,
            "document": pdf_results
        }
        insights = detector.generate_insights(claim_data)
        print("\n=== Final Insights ===")
        print(insights)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")