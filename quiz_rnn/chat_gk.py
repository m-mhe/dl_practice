import json
import re

import torch
import torch.nn as nn


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

MODEL_PATH = "quiz_ai.pth"
VOCAB_PATH = "vocabulary.json"

EMBEDDING_SIZE = 64
HIDDEN_NEURONS = 128

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# --------------------------------------------------
# 2. Tokenizer
# --------------------------------------------------

def tokenizer(text):

    text = re.sub(r"[^a-z0-9\s]", "", text.lower())
    text = re.sub(r"\s+", " ", text)

    return text.split()


# --------------------------------------------------
# 3. Model architecture
# --------------------------------------------------

class QuizModel(nn.Module):
    def __init__(
        self,
        vocabSize,
        embeddingSize=64,
        hiddenNeurons=128
    ):
        super().__init__()

        self.embed = nn.Embedding(
            num_embeddings=vocabSize,
            embedding_dim=embeddingSize
        )

        self.rnn = nn.RNN(
            input_size=embeddingSize,
            hidden_size=hiddenNeurons,
            batch_first=True
        )

        self.net = nn.Sequential(
            nn.Linear(hiddenNeurons, hiddenNeurons),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(hiddenNeurons, hiddenNeurons),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(hiddenNeurons, vocabSize)
        )

    def forward(self, X):
        X = self.embed(X)
        _, hidden = self.rnn(X)

        return self.net(hidden.squeeze(0))


# --------------------------------------------------
# 4. Load vocabulary
# --------------------------------------------------

try:
    with open(VOCAB_PATH, "r", encoding="utf-8") as file:
        vocabulary = json.load(file)

except FileNotFoundError:
    raise SystemExit(f"Vocabulary file not found: {VOCAB_PATH}")

if "<UNK>" not in vocabulary:
    raise SystemExit("The vocabulary must contain the '<UNK>' token.")

idx_to_token = {
    index: token
    for token, index in vocabulary.items()
}

vocab_size = len(vocabulary)
unknown_index = vocabulary["<UNK>"]


# --------------------------------------------------
# 5. Load trained model
# --------------------------------------------------

model = QuizModel(
    vocabSize=vocab_size,
    embeddingSize=EMBEDDING_SIZE,
    hiddenNeurons=HIDDEN_NEURONS
).to(device)

try:
    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
    model.load_state_dict(state_dict)

except FileNotFoundError:
    raise SystemExit(f"Model file not found: {MODEL_PATH}")

except RuntimeError as error:
    raise SystemExit(
        "Could not load model weights. Check that the model "
        "architecture and vocabulary match those used for training.\n"
        f"Details: {error}"
    )

model.eval()

print("Model loaded successfully!")
print("Enter a question, or type 'exit' to quit.")
print("-" * 50)


# --------------------------------------------------
# 6. Predict an answer
# --------------------------------------------------

def predict_answer(question):
    tokens = tokenizer(question)

    if not tokens:
        return "<no input>"

    # Convert tokens into vocabulary indices.
    token_ids = [
        vocabulary.get(token, unknown_index)
        for token in tokens
    ]

    # Add batch dimension: (sequence_length,)
    # becomes (1, sequence_length).
    X = torch.tensor(
        token_ids,
        dtype=torch.long
    ).unsqueeze(0).to(device)

    # Inference mode disables gradient calculation.
    with torch.inference_mode():
        logits = model(X)

        predicted_index = logits.argmax(dim=1).item()

    return idx_to_token.get(predicted_index, "<UNK>")


# --------------------------------------------------
# 7. Interactive question loop
# --------------------------------------------------

while True:
    try:
        question = input("\nQuestion: ").strip()

    except (EOFError, KeyboardInterrupt):
        print("\nExiting...")
        break

    if question.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break

    if not question:
        print("Please enter a question.")
        continue

    answer = predict_answer(question)

    print(f"AI answer: {answer.capitalize()}")