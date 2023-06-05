from PIL import Image
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

# Load pre-trained models
mtcnn = MTCNN(image_size=160)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Function to extract face embeddings
def get_face_embeddings(image):
    # Convert image to numpy array
    image_np = np.array(image)

    # Add channel dimension
    image_np = np.expand_dims(image_np, axis=0)

    # Detect faces
    boxes, _ = mtcnn.detect(image_np)

    # Check if a face was detected
    if boxes is not None and len(boxes) > 0:
        # Convert image to tensor
        image_tensor = torch.tensor(image_np).permute(0, 3, 1, 2).float()

        # Get face embeddings
        embeddings = resnet(image_tensor)
        return embeddings
    else:
        return None

# Load the two photos to compare
photo1 = Image.open('2.png')
photo2 = Image.open('2-2.jpg')

# Extract face embeddings
embedding1 = get_face_embeddings(photo1)
embedding2 = get_face_embeddings(photo2)

# Check if face embeddings were extracted successfully
if embedding1 is not None and embedding2 is not None:
    # Calculate Euclidean distance between embeddings
    distance = torch.nn.functional.pairwise_distance(embedding1, embedding2).item()

    # Set a threshold for similarity
    similarity_threshold = 0.6

    # Compare the distance with the similarity threshold
    if distance < similarity_threshold:
        print("The photos are the same.")
    else:
        print("The photos are different.")
else:
    print("No face detected in one or both of the photos.")
