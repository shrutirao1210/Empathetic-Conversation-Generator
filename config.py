import torch

# Phase 1: Emotion Classifier
ROBERTA_MODEL_NAME = "roberta-base"

# Phase 2: Generator Context
BART_MODEL_NAME = "facebook/bart-base"

# GoEmotions Classes (28 classes including neutral)
GO_EMOTIONS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

NUM_EMOTIONS = len(GO_EMOTIONS)
EMOTION_TO_LABEL = {emotion: idx for idx, emotion in enumerate(GO_EMOTIONS)}
LABEL_TO_EMOTION = {idx: emotion for idx, emotion in enumerate(GO_EMOTIONS)}
NEUTRAL_IDX = EMOTION_TO_LABEL['neutral']

# Architecture & Training Hyperparameters
MAX_LEN = 128
BATCH_SIZE = 16
LEARNING_RATE_PHASE1_BASE = 2e-5
LEARNING_RATE_PHASE1_HEAD = 1e-4
LEARNING_RATE_PHASE2 = 3e-5
EPOCHS_PHASE1 = 4
EPOCHS_PHASE2 = 4

# Confidence Gate Threshold - ADJUSTED FOR 28 CLASSES
CONFIDENCE_THRESHOLD = 0.55

# Inference settings
BEAM_SIZE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
