import cv2
import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd
import json
import io
import os
from typing import Dict, List, Tuple

# Define answer keys for different versions
ANSWER_KEYS = {
    "SET A": {
        "Python": {1: "a", 2: "c", 3: "c", 4: "c", 5: "c", 6: "a", 7: "c", 8: "c", 9: "b", 10: "c",
                  11: "a", 12: "a", 13: "d", 14: "a", 15: "b", 16: "a,b,c,d", 17: "c", 18: "d", 19: "a", 20: "b"},
        "EDA": {21: "a", 22: "d", 23: "b", 24: "a", 25: "c", 26: "b", 27: "a", 28: "b", 29: "d", 30: "c",
               31: "c", 32: "a", 33: "b", 34: "c", 35: "a", 36: "b", 37: "d", 38: "b", 39: "a", 40: "b"},
        "SQL": {41: "c", 42: "c", 43: "c", 44: "b", 45: "b", 46: "a", 47: "c", 48: "b", 49: "d", 50: "a",
               51: "c", 52: "b", 53: "c", 54: "c", 55: "a", 56: "b", 57: "b", 58: "a", 59: "a,b", 60: "b"},
        "POWER BI": {61: "b", 62: "c", 63: "a", 64: "b", 65: "c", 66: "b", 67: "b", 68: "c", 69: "c", 70: "b",
                    71: "b", 72: "b", 73: "d", 74: "b", 75: "a", 76: "b", 77: "b", 78: "b", 79: "b", 80: "b"},
        "Statistics": {81: "a", 82: "b", 83: "c", 84: "b", 85: "c", 86: "b", 87: "b", 88: "b", 89: "a", 90: "b",
                      91: "c", 92: "b", 93: "c", 94: "b", 95: "b", 96: "b", 97: "c", 98: "a", 99: "b", 100: "c"}
    },
    "SET B": {
        "Python": {1: "a", 2: "b", 3: "d", 4: "b", 5: "b", 6: "d", 7: "c", 8: "c", 9: "a", 10: "c",
                  11: "a", 12: "b", 13: "d", 14: "c", 15: "c", 16: "a", 17: "c", 18: "b", 19: "d", 20: "c"},
        "EDA": {21: "a", 22: "a", 23: "b", 24: "a", 25: "b", 26: "a", 27: "b", 28: "b", 29: "c", 30: "c",
               31: "b", 32: "c", 33: "b", 34: "c", 35: "a", 36: "a", 37: "a", 38: "b", 39: "b", 40: "a"},
        "SQL": {41: "b", 42: "a", 43: "d", 44: "b", 45: "c", 46: "b", 47: "b", 48: "b", 49: "b", 50: "b",
               51: "c", 52: "a", 53: "c", 54: "a", 55: "c", 56: "c", 57: "b", 58: "a", 59: "b", 60: "c"},
        "POWER BI": {61: "b", 62: "b", 63: "b", 64: "d", 65: "c", 66: "b", 67: "b", 68: "a", 69: "b", 70: "b",
                    71: "b", 72: "c", 73: "a", 74: "d", 75: "b", 76: "b", 77: "d", 78: "a", 79: "b", 80: "a"},
        "Statistics": {81: "b", 82: "c", 83: "b", 84: "a", 85: "c", 86: "b", 87: "b", 88: "a", 89: "b", 90: "d",
                      91: "c", 92: "d", 93: "b", 94: "b", 95: "b", 96: "c", 97: "c", 98: "b", 99: "b", 100: "c"}
    }
}

class OMREvaluator:
    def __init__(self):
        self.answer_keys = ANSWER_KEYS
        self.subjects = ["Python", "EDA", "SQL", "POWER BI", "Statistics"]
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess the OMR sheet image"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        return thresh
    
    def find_omr_sheet(self, image: np.ndarray) -> np.ndarray:
        """Find and extract the OMR sheet from the image"""
        # Preprocess the image
        processed = self.preprocess_image(image)
        
        # Find contours
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour (assuming it's the OMR sheet)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Crop the image to the OMR sheet
        cropped = image[y:y+h, x:x+w]
        
        return cropped
    
    def detect_bubbles(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect bubbles in the OMR sheet"""
        # Preprocess the image
        processed = self.preprocess_image(image)
        
        # Find contours
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bubbles = []
        for contour in contours:
            # Filter contours by area (to exclude small noise)
            area = cv2.contourArea(contour)
            if 100 < area < 1000:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                bubbles.append((x, y, w, h))
        
        return bubbles
    
    def extract_answers(self, image: np.ndarray, bubbles: List[Tuple[int, int, int, int]]) -> Dict[int, str]:
        """Extract answers from detected bubbles"""
        # Preprocess the image
        processed = self.preprocess_image(image)
        
        answers = {}
        
        # Sort bubbles by position (assuming a grid layout)
        bubbles.sort(key=lambda b: (b[1], b[0]))  # Sort by y then x
        
        # Group bubbles by row (question)
        current_y = bubbles[0][1]
        row_bubbles = []
        rows = []
        
        for bubble in bubbles:
            x, y, w, h = bubble
            if abs(y - current_y) > 10:  # New row
                rows.append(row_bubbles)
                row_bubbles = []
                current_y = y
            row_bubbles.append(bubble)
        rows.append(row_bubbles)
        
        # For each row, determine which bubble is filled
        for i, row in enumerate(rows):
            row.sort(key=lambda b: b[0])  # Sort by x position
            
            # Calculate the average intensity of each bubble
            intensities = []
            for bubble in row:
                x, y, w, h = bubble
                roi = processed[y:y+h, x:x+w]
                intensity = np.mean(roi)
                intensities.append(intensity)
            
            # The bubble with the highest intensity is the filled one
            if intensities:
                max_idx = np.argmax(intensities)
                answer_letter = chr(ord('a') + max_idx)
                answers[i+1] = answer_letter
        
        return answers
    
    def evaluate_answers(self, extracted_answers: Dict[int, str], answer_key: Dict[str, Dict[int, str]]) -> Dict[str, Dict]:
        """Evaluate extracted answers against the answer key"""
        results = {}
        total_score = 0
        
        for subject in self.subjects:
            subject_answers = answer_key[subject]
            subject_score = 0
            subject_results = {}
            
            for q_num, correct_answer in subject_answers.items():
                if q_num in extracted_answers:
                    student_answer = extracted_answers[q_num]
                    is_correct = student_answer == correct_answer
                    subject_results[q_num] = {
                        "correct": is_correct,
                        "student_answer": student_answer,
                        "correct_answer": correct_answer
                    }
                    if is_correct:
                        subject_score += 1
                else:
                    subject_results[q_num] = {
                        "correct": False,
                        "student_answer": "N/A",
                        "correct_answer": correct_answer
                    }
            
            results[subject] = {
                "score": subject_score,
                "details": subject_results
            }
            total_score += subject_score
        
        results["total_score"] = total_score
        return results
    
    def process_omr_sheet(self, image: np.ndarray, set_version: str) -> Dict:
        """Process an OMR sheet and return results"""
        # Find and extract the OMR sheet
        omr_sheet = self.find_omr_sheet(image)
        
        # Detect bubbles
        bubbles = self.detect_bubbles(omr_sheet)
        
        # Extract answers
        extracted_answers = self.extract_answers(omr_sheet, bubbles)
        
        # Evaluate answers
        results = self.evaluate_answers(extracted_answers, self.answer_keys[set_version])
        
        return results

# Streamlit web application
def main():
    st.set_page_config(page_title="OMR Evaluation System", page_icon="📝", layout="wide")
    
    st.title("Automated OMR Evaluation & Scoring System")
    st.markdown("Upload images of OMR sheets to evaluate and score them automatically.")
    
    # Initialize evaluator
    evaluator = OMREvaluator()
    
    # File upload
    uploaded_files = st.file_uploader("Upload OMR sheet images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    # Set version selection
    set_version = st.selectbox("Select OMR Sheet Version", list(ANSWER_KEYS.keys()))
    
    if uploaded_files:
        results = []
        
        for uploaded_file in uploaded_files:
            # Read image
            image = Image.open(uploaded_file)
            image_np = np.array(image.convert('RGB'))
            
            # Process OMR sheet
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    result = evaluator.process_omr_sheet(image_np, set_version)
                    results.append({
                        "filename": uploaded_file.name,
                        "results": result
                    })
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        # Display results
        if results:
            st.subheader("Evaluation Results")
            
            # Create a summary table
            summary_data = []
            for result in results:
                row = {"Filename": result["filename"]}
                for subject in evaluator.subjects:
                    row[subject] = result["results"][subject]["score"]
                row["Total Score"] = result["results"]["total_score"]
                summary_data.append(row)
            
            df = pd.DataFrame(summary_data)
            st.dataframe(df)
            
            # Detailed results for each file
            for result in results:
                with st.expander(f"Detailed results for {result['filename']}"):
                    st.write(f"**Total Score:** {result['results']['total_score']}/100")
                    
                    for subject in evaluator.subjects:
                        st.write(f"**{subject}:** {result['results'][subject]['score']}/20")
                    
                    # Show subject-wise details
                    for subject in evaluator.subjects:
                        st.subheader(f"{subject} Details")
                        subject_details = result['results'][subject]['details']
                        
                        details_data = []
                        for q_num, details in subject_details.items():
                            details_data.append({
                                "Question": q_num,
                                "Student Answer": details['student_answer'],
                                "Correct Answer": details['correct_answer'],
                                "Correct": "Yes" if details['correct'] else "No"
                            })
                        
                        details_df = pd.DataFrame(details_data)
                        st.dataframe(details_df)
            
            # Download results as CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="omr_results.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()