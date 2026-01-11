import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length, when, size, split, lower

# ==========================================
# PART 1: BIG DATA PROCESSING (PySpark)
# ==========================================
print("--- Step 1: Initialize Spark Session ---")
# This 'builder' pattern is how you start any Spark application
spark = SparkSession.builder.appName("SpamDetector").getOrCreate()

# 1. Create Dummy Data (In a real job, this would be a CSV load)
# 0 = Ham (Good), 1 = Spam (Bad)
data = [
    ("Hi mom, how are you?", 0),
    ("WIN A FREE IPHONE NOW CLICK HERE", 1),
    ("Meeting at 3pm today", 0),
    ("CONGRATULATIONS YOU WON MILLIONS", 1),
    ("Can you review this code?", 0),
    ("Urgent wire transfer needed immediately", 1),
    ("Lunch tomorrow?", 0),
    ("FREE FREE FREE MONEY", 1)
]
columns = ["text", "label"]

# Create the Distributed DataFrame
df = spark.createDataFrame(data, columns)

print("--- Step 2: Feature Engineering with Spark ---")
# We use Spark to process the text into numbers (features) the AI can understand.
# Feature A: Length of the message
# Feature B: Does it contain the word "free"? (1 if yes, 0 if no)
# Feature C: Is it in all caps? (Approximated by checking if upper version equals original)

processed_df = df.withColumn("length", length(col("text"))) \
                 .withColumn("has_free", when(col("text").contains("FREE"), 1).otherwise(0)) \
                 .withColumn("is_shouting", when(col("text") == fl_upper := split(col("text"), " ")[0], 1).otherwise(0)) 
                 # (Simplified 'shouting' logic for demo purposes: checks strictly if text is uppercase is tricky in simple regex, 
                 # so we will stick to a simpler logic for this demo:
                 # Let's just use "length" and "has_free" for simplicity, and add "word_count")

# RE-DOING Feature Engineering for clarity and reliability:
processed_df = df.select(
    col("label"),
    length(col("text")).alias("char_count"),
    size(split(col("text"), " ")).alias("word_count"), # This counts spaces to guess word count
    when(lower(col("text")).contains("free"), 1.0).otherwise(0.0).alias("has_free_keyword")
)

print("Processed Spark Data (First 5 rows):")
processed_df.show()

# Convert to Pandas/Numpy for PyTorch
# In Big Data, you use Spark to shrink petabytes of text down to these numbers, 
# then collect the small number matrix to the driver for training.
pandas_df = processed_df.toPandas()
X_numpy = pandas_df[["char_count", "word_count", "has_free_keyword"]].values.astype(np.float32)
y_numpy = pandas_df["label"].values.astype(np.float32).reshape(-1, 1)

# Stop Spark to free up resources
spark.stop()


# ==========================================
# PART 2: MACHINE LEARNING (PyTorch)
# ==========================================
print("\n--- Step 3: Define PyTorch Model ---")

# Convert numpy arrays to PyTorch Tensors
X_tensor = torch.from_numpy(X_numpy)
y_tensor = torch.from_numpy(y_numpy)

# Define a simple Neural Network
class SpamClassifier(nn.Module):
    def __init__(self):
        super(SpamClassifier, self).__init__()
        # Input: 3 features (char_count, word_count, has_free_keyword)
        # Output: 1 prediction (Probability of being spam)
        self.layer1 = nn.Linear(3, 5) # Hidden layer
        self.relu = nn.ReLU()         # Activation function
        self.layer2 = nn.Linear(5, 1) # Output layer
        self.sigmoid = nn.Sigmoid()   # Squishes result between 0 and 1

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return self.sigmoid(x)

model = SpamClassifier()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss() # Binary Cross Entropy Loss (Standard for Yes/No classification)

print("--- Step 4: Train the Model ---")
epochs = 100
for epoch in range(epochs):
    # Forward pass (Make a guess)
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    
    # Backward pass (Calculate error and update weights)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

print("\n--- Step 5: Test Prediction ---")
# Let's test a fake email: "FREE MONEY" 
# Features: [Length=10, Words=2, Has_Free=1]
test_email = torch.tensor([[10.0, 2.0, 1.0]])
prediction = model(test_email).item()

print(f"Prediction for 'FREE MONEY': {prediction:.4f}")
print("Note: Closer to 1.0 means SPAM, Closer to 0.0 means HAM")
