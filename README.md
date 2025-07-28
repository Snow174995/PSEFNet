# PSEFNet
PSEF-Net: A Position-Aware Sentiment Enhancement Framework for Student Performance and Teaching Quality Understanding

Overview
PSEF-Net is a deep learning model designed for student performance prediction based on online course comment data. It enhances the interpretability and accuracy of predictions by incorporating position-aware sentiment modeling and structural information fusion. The model is suitable for applications in educational quality assessment and personalized learning analytics.

Key Features
- **Position-aware Sentiment Enhancement**: Captures sentiment polarity and its structural position in student comments.
- **Cross-attention Structure Fusion**: Fuses emotional features with structural course information.
- **Contextual Structure Modeling**: Employs convolution and Transformer layers for semantic aggregation and structure-sensitive reasoning.
- **Support for Dual-task Extension**: Can be extended to support auxiliary sentiment supervision.

Environment Requirements
- Python >= 3.8
- PyTorch >= 1.10
- pandas
- numpy
- matplotlib

Install dependencies via pip:
"bash pip install torch pandas numpy matplotlib"

Dataset Preparation
The input CSV files "Mooc_dataset.csv" and "Netease_dataset.csv" should include the following fields:
comment_text: Textual comment from student
grade_score: Normalized test score (target label)
sentiment_label: Sentiment polarity label (-1, 0, 1)
course_rating: Likert scale (1–5) for course satisfaction
course_id: Identifier of the course

Model Description
PSEFNet contains three core modules:
- **PSEModule**: Applies position-aware attention over emotional cues.
- **EmotionStructureFusion**: Uses query-key attention to integrate structural embeddings.
- **StructureContextFusion**: Combines convolutional and Transformer encoders for high-level contextual modeling.

Inputs
x_embed: Comment embedding (B, L, D)
structure_vec: Structural vector (B, D)

Output
Predicted grade score (B, 1)

Training & Evaluation
To train and evaluate the model, run:
"bash python PSEFNet.py"


Key Hyperparameters in "PSEF-Net":

-input_window_size: Input sequence length (default 12)
- output_window_size: Prediction horizon (default 12)
- hidden_dim: Hidden layer dimension (default 128)
- dropout: Dropout rate (default 0.1)

Visualization
The system outputs evaluation metrics and plots predicted vs actual scores for analysis. Predicted results are also saved to CSV for further inspection.
