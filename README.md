# Decoupled Empathetic Chatbot

An emotionally-aware conversational AI system built using a **decoupled two-stage transformer architecture** for **emotion detection** and **empathetic response generation**.

The project combines **RoBERTa** for emotion classification and **BART-base** for conversational response generation using a custom **Emotion-Gated Cross-Attention Fusion Layer** that injects emotional understanding directly into the response generation process.

Unlike traditional chatbot architectures that use a single shared model for both emotion understanding and response generation, this system separates both tasks into independent modules. This avoids optimization conflicts between classification and generation, resulting in more accurate emotional understanding and more contextually appropriate responses.

---

# Project Overview

The objective of this project is to develop a chatbot capable of understanding the emotional state of a user and generating emotionally appropriate conversational responses.

The system operates in two major stages:

1. **Emotion Understanding**
   - The user's input is analyzed using a fine-tuned RoBERTa model
   - The model predicts emotional probabilities across 28 GoEmotions classes
   - A confidence-based mechanism handles uncertain predictions

2. **Empathetic Response Generation**
   - Emotional information is fused with conversational context
   - A BART-based generator produces contextually relevant and emotionally aligned responses
   - Multi-turn conversational history is preserved for coherent dialogue generation

The architecture uses:
- Soft emotion embeddings
- Emotion-Gated Cross-Attention
- Multi-turn contextual memory
- Beam search and nucleus sampling
- Emotion-conditioned decoding

to generate responses that feel more natural, emotionally aware, and conversationally coherent.

---

# Motivation

Traditional conversational AI systems are often capable of generating grammatically correct responses but fail to understand emotional nuance.

For example:

```text
User: I feel exhausted and overwhelmed lately.

Generic Bot:
"Okay. Tell me more."

Empathetic Bot:
"That sounds really stressful. Have things been especially difficult recently?"
```

Emotionally intelligent dialogue systems require:
- Emotion recognition
- Context understanding
- Emotionally conditioned response generation

Most existing architectures attempt to optimize all these tasks using a single shared encoder, which creates conflicts between:
- Emotion classification objectives
- Natural language generation objectives

This project addresses that issue using a **decoupled architecture** where separate transformer models specialize in separate tasks.

---

# System Architecture

```text
User Input
   │
   ▼
RoBERTa Emotion Classifier
   │
   ├── Emotion Label
   ├── Confidence Score
   ▼
Confidence Gate
   │
   ▼
Soft Emotion Embedding
   │
   ▼
Emotion-Gated Cross-Attention
   │
   ▼
BART Encoder-Decoder
   │
   ▼
Empathetic Response
```

---

# Workflow

## Step 1 — User Input

The chatbot receives a natural language sentence from the user.

Example:

```text
"I have been feeling really overwhelmed with work lately."
```

---

## Step 2 — Emotion Detection using RoBERTa

The input sentence is passed into a fine-tuned **RoBERTa** encoder trained on the GoEmotions dataset.

The classifier predicts probabilities across 28 emotion categories such as:
- sadness
- anger
- fear
- excitement
- admiration
- nervousness
- grief
- neutral

Example output:

```text
sadness      → 0.51
nervousness  → 0.28
fear         → 0.11
```

The model outputs:
- Predicted emotion
- Confidence score
- Full emotion probability distribution

---

# Confidence Gate

Human emotions are often ambiguous.

Instead of forcing unreliable predictions, the system uses a confidence threshold.

If the confidence score is too low:
- The model routes the input to a neutral emotional representation
- This prevents unstable or emotionally incorrect responses

This improves robustness during real-world conversations.

---

# Soft Emotion Embedding

Instead of selecting only one hard emotion label, the system computes a weighted emotional representation using the full probability distribution.

Example:

```text
sadness      → 0.51
nervousness  → 0.28
fear         → 0.11
```

The final emotional embedding becomes a blend of multiple emotions.

This helps the chatbot:
- Handle emotional ambiguity
- Produce smoother emotional transitions
- Generate more nuanced responses

---
# Emotion Mapping Strategy

The emotion classifier and response generator are trained on two different datasets:

| Dataset | Purpose |
|---|---|
| GoEmotions | Emotion classification |
| EmpatheticDialogues | Response generation |

Since both datasets use different emotion label spaces, an intermediate mapping layer is used.

The RoBERTa classifier predicts one of the 28 GoEmotions categories, which are then mapped to emotionally similar EmpatheticDialogues tags before response generation.

Examples:

| GoEmotions Label | Mapped ED Emotion |
|---|---|
| grief | devastated |
| sadness | sad |
| excitement | excited |
| admiration | proud |
| nervousness | anxious |
| annoyance | angry |
| fear | afraid |
| joy | joyful |

This mapping ensures that the emotional representation used during inference matches the emotional context distribution seen by BART during training on EmpatheticDialogues.

The mapped emotion tag is prepended to the conversational context before being passed into the BART encoder.

Example:

```text
[sad]
User: I feel really lonely lately.
```

This helps the generator produce responses aligned with the detected emotional state.

# Emotion-Gated Cross-Attention

This is the core contribution of the project.

The emotional embedding is fused with conversational context using:
- Multi-head attention
- Learnable gating
- Residual fusion

The gating mechanism allows the model to determine:

> Which words or contextual regions should receive stronger emotional emphasis?

For example:
- emotionally important words like *"alone"* or *"overwhelmed"* receive stronger emotional modulation
- neutral filler words receive less influence

This produces emotionally conditioned contextual representations before decoding.

---

# Response Generation using BART

The final stage uses `facebook/bart-base` for empathetic response generation.

The generator receives:
- Conversation history
- Emotional embeddings
- Contextual encoder representations

Input format:

```text
[sadness]
User: I feel exhausted lately.
Bot: I'm sorry you're going through that.
User: It has been getting worse recently.
```

The decoder then generates emotionally aligned conversational responses.

Generation uses:
- Beam search
- Nucleus sampling
- Temperature scaling
- Emotion-conditioned prompts

to produce responses that are:
- more diverse
- more natural
- less repetitive

---

# Why Decoupled Architecture?

Traditional chatbot architectures often use a single shared encoder for:
- emotion classification
- response generation

This creates optimization conflicts because:
- classification requires highly discriminative sentence representations
- generation requires contextual sequence representations

A shared encoder must compromise between these objectives.

This project eliminates that issue by separating the system into two specialized transformer models:

| Component | Responsibility |
|---|---|
| RoBERTa | Emotion Understanding |
| BART | Conversational Generation |

Benefits of this approach:
- Better emotion recognition
- More stable training
- Improved conversational quality
- Cleaner modular design
- Easier experimentation

---

# Datasets Used

## GoEmotions

Used for emotion classification training.

Features:
- ~58k Reddit comments
- 28 emotion categories
- Fine-grained emotional annotations

---

## EmpatheticDialogues

Used for response generation training.

Features:
- ~76k empathetic conversations
- Human-to-human dialogues
- Emotion-grounded responses
- Multi-turn conversational structure

---

# Model Components

| Component | Purpose |
|---|---|
| RoBERTa | Emotion classification |
| Emotion Classifier Head | Predict emotion probabilities |
| Confidence Gate | Handle uncertain predictions |
| Soft Emotion Embedding | Represent blended emotions |
| Cross-Attention Fusion | Inject emotional context |
| BART | Response generation |

---

# Training Strategy

The system is trained in two separate phases.

## Phase 1 — Emotion Classification

Train:
- RoBERTa encoder
- Classification head

Dataset:
- GoEmotions

Objective:
- Learn emotion-discriminative representations

---

## Phase 2 — Response Generation

Train:
- BART
- Fusion layer
- Emotion embeddings

Freeze:
- RoBERTa classifier

Dataset:
- EmpatheticDialogues

Objective:
- Learn emotionally conditioned conversational generation

---

# Evaluation Results

| Metric | Score |
|---|---|
| BLEU-4 | 2.14 |
| Perplexity | 17.42 |
| Emotion Classes | 28 |
| Training Phases | 2 |

The results fall within expected ranges for open-domain empathetic dialogue systems.

---

# Example Conversations

## Positive Emotion

```text
User: I just got accepted into my dream college!

Bot: Congratulations! That must feel amazing. What are you planning to study?
```

---

## Stress / Anxiety Example

```text
User: I feel overwhelmed with work lately.

Bot: That sounds really stressful. Do you want to talk about what has been happening?
```

---

## Negative Emotion

```text
User: My roommate keeps eating my food and it's really frustrating.

Bot: That sounds really annoying. Have you talked to them about it yet?
```

---

# Key Features

- Decoupled transformer architecture
- Emotion-aware conversational generation
- RoBERTa-based emotion classification
- BART-based response generation
- Emotion-Gated Cross-Attention fusion
- Confidence-aware fallback routing
- Soft emotion embedding representation
- Multi-turn conversational memory
- Beam search + nucleus sampling decoding
- Emotion-conditioned response generation

---

## Model Checkpoints

The trained model checkpoints for this project are available on Hugging Face:

🔗 https://huggingface.co/shrutimrao/empathetic-chatbot

### Available Files
- `best_decoupled_model.pt` — Main decoupled empathetic dialogue generation model
- `best_roberta_head.pt` — Emotion classification head based on RoBERTa

These checkpoints can be used for:
- Inference
- Fine-tuning
- Emotion-aware response generation
- Reproducing experimental results

# Future Improvements

Potential future enhancements include:
- Upgrading to `bart-large`
- RLHF-based empathy optimization
- Commonsense reasoning integration
- Long-term conversational memory
- Persona consistency
- Retrieval-augmented generation
- Emotion smoothing across conversation turns

---

# Limitations

Current limitations include:
- Responses may occasionally become generic
- Limited factual knowledge handling
- Emotion ambiguity can trigger fallback routing
- Long-term reasoning remains limited by model size

---

# References

- GoEmotions Dataset
- EmpatheticDialogues Dataset
- Hugging Face Transformers
- RoBERTa
- BART
- PyTorch
