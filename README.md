
# YouTube Comment Sentiment Analysis Tool

![image](https://github.com/RanaweeraHK/Youtube-Comment-Sentiment-Analyzer/assets/129282753/88d24baf-c500-4fa1-9f2c-48d14b0f9a5f)


## Overview

This project is a YouTube Comment Sentiment Analysis Tool that fetches comments from a YouTube video, processes them using natural language processing (NLP) techniques, performs sentiment analysis, and visualizes the sentiment distribution in a pie chart. The backend is built with Flask, while the frontend is developed using React for a seamless user experience.

## Key Features

- **Data Collection**: Utilizes the YouTube API to fetch comments from any YouTube video.
- **Text Preprocessing**: Removes stop words and punctuation for cleaner analysis using Spacy.
- **Sentiment Analysis**: Utilizes a pre-trained BERT-based model from Hugging Face for sentiment classification.
- **Data Visualization**: Generates a pie chart using Matplotlib to illustrate the sentiment distribution of comments.
- **Sleek User Interface**: The React frontend provides an intuitive and interactive user interface for easy usage.

## Technology Stack

- **Backend**: Flask, Transformers, Spacy
- **Frontend**: React
- **Sentiment Analysis Model**: BERT-based model from Hugging Face Transformers library
- **Data Fetching**: YouTube API
- **Data Visualization**: Matplotlib

## Workflow

1. **Input**: Provide a YouTube video URL.
2. **Processing**: Extract video ID, fetch comments, preprocess text, and analyze sentiment.
3. **Output**: Visualize the sentiment distribution with a pie chart.

## Getting Started

### Prerequisites

- Python 3.x
- Node.js
- YouTube API key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   ```
2. Install backend dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
3. Install frontend dependencies:
   ```bash
   cd frontend
   npm install
   ```

### Usage

1. Start the Flask backend server:
   ```bash
   cd backend
   python app.py
   ```
2. Start the React frontend:
   ```bash
   cd frontend
   npm start
   ```
3. Open your browser and navigate to `http://localhost:3000` to access the tool.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Flask](https://flask.palletsprojects.com/)
- [React](https://reactjs.org/)
- [YouTube API](https://developers.google.com/youtube)
