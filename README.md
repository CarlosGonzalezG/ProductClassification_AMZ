# Solution designed and implemented by Carlos González
I have implemented the solution in a short time, fulfilling the described requirements to address the identified problem of an unbalanced dataset. Some classes, such as **Health & Personal Care**, had few samples, so I adjusted the weights to be taken into account during training. Additionally, I implemented the code to allow multiple models to be trained from a single configuration file. The models I have tested are:

- **Logistic Regression**
- **XGBoost**
- **CatBoost**
- **LightGBM**

For the boosting algorithms, I reviewed the following paper: [https://arxiv.org/abs/2305.17094](https://arxiv.org/abs/2305.17094).

The classification results could be found inside each model folder with a .txt file.

## Dockerization

I have dockerized both the training and inference parts of the project.

### Training

**IMPORTANT NOTE: Add embeddings folder inside model/ to avoid the recalculation of embeddings.**

To set up the training environment, execute the following commands:

```bash
docker build -t product_class_train -f train/Dockerfile.train .
```

```bash
docker run --rm -v $(pwd)/model:/app/model product_class_train
```

### Inference
To set up the training environment, execute the following commands:

```bash
docker build -t fastapi-api -f inference/Dockerfile.inference .
```

```bash
docker run -d -p 8000:8000 fastapi-api  
```

In any browser locally you can access the endpoint: http://127.0.0.1:8000/docs, then with the ‘Try it out’ button and filling in the fields that appear in the template and hitting the ‘Execute’ button you can make inference on the model previously configured in the config_inference.yaml.

# Limitations
Due to being a test to be carried out in a short time, it has been decided to leave the hyperparameter search with GridSearch pending, but a broad knowledge of the proposed challenge has been demonstrated.

# Future works
As future work to improve the performance of the model I propose several things, analyse in detail the importance of each input feature of the model and see how they affect the performance of the model, do cross validation, apply an open-source OCR to try to add a description of the product images or even try to train a VisionTransformer, I can also think of doing a fine-tunning of the sentence transformers using SetFit with a specific dataset of products from online shops.

---
# Product Classification

In this test we ask you to build a model to classify products into their categories according to their features.

## Dataset Description

The dataset is a simplified version of [Amazon 2018](https://jmcauley.ucsd.edu/data/amazon/), only containing products and their descriptions.

The dataset consists of a jsonl file where each is a json string describing a product.

Example of a product in the dataset:
```json
{
 "also_buy": ["B071WSK6R8", "B006K8N5WQ", "B01ASDJLX0", "B00658TPYI"],
 "also_view": [],
 "asin": "B00N31IGPO",
 "brand": "Speed Dealer Customs",
 "category": ["Automotive", "Replacement Parts", "Shocks, Struts & Suspension", "Tie Rod Ends & Parts", "Tie Rod Ends"],
 "description": ["Universal heim joint tie rod weld in tube adapter bung. Made in the USA by Speed Dealer Customs. Tube adapter measurements are as in the title, please contact us about any questions you 
may have."],
 "feature": ["Completely CNC machined 1045 Steel", "Single RH Tube Adapter", "Thread: 3/4-16", "O.D.: 1-1/4", "Fits 1-1/4\" tube with .120\" wall thickness"],
 "image": [],
 "price": "",
 "title": "3/4-16 RH Weld In Threaded Heim Joint Tube Adapter Bung for 1-1/4&quot; Dia by .120 Wall Tube",
 "main_cat": "Automotive"
}
```

### Field description
- also_buy/also_view: IDs of related products
- asin: ID of the product
- brand: brand of the product
- category: list of categories the product belong to, usually in hierarchical order
- description: description of the product
- feature: bullet point format features of the product
- image: url of product images (migth be empty)
- price: price in US dollars (might be empty)
- title: name of the product
- main_cat: main category of the product

`main_cat` can have one of the following values:
```json
["All Electronics",
 "Amazon Fashion",
 "Amazon Home",
 "Arts, Crafts & Sewing",
 "Automotive",
 "Books",
 "Buy a Kindle",
 "Camera & Photo",
 "Cell Phones & Accessories",
 "Computers",
 "Digital Music",
 "Grocery",
 "Health & Personal Care",
 "Home Audio & Theater",
 "Industrial & Scientific",
 "Movies & TV",
 "Musical Instruments",
 "Office Products",
 "Pet Supplies",
 "Sports & Outdoors",
 "Tools & Home Improvement",
 "Toys & Games",
 "Video Games"]
```

[Download dataset](https://drive.google.com/file/d/1Zf0Kdby-FHLdNXatMP0AD2nY0h-cjas3/view?usp=sharing)

Data can be read directly from the gzip file as:
```python
def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.loads(l)
```

## Task description

- You should create a model that predicts `main_cat` using any of the other fields except `category` list. The model should be developed in python, you can use any pre-trained models and thirdparty 
libraries you need (for example huggingface).

- You should create a HTTP API endpoint that is capable of performing inference and return the predicted `main_cat` when receiving the rest of product fields.

- Both the training code (if needed) and the inference API should be dockerized and easy for us to run and test locally. **Only docker build and docker run commands should be necessary to perform training or setting up the inference API**.

- You should also provide a detailed analysis of the performance of your model. **In this test we're not looking for the best model performance but we expect a good understanding of your solution performance and it's limitations**.

- We will value:
    - Code cleanliness and good practices.
    - API design and architecture.
    - A clear understanding of the model performance, strengths and weak points.
---

- Answer the following questions:

        - What would you change in your solution if you needed to predict all the categories?

Currently, the model predicts only a single label—namely the primary category (main_cat). If you needed to predict all categories (for example, a list of categories or hierarchical labels), you would move from single-label classification to multi-label or hierarchical classification.

For example if each product may belong to multiple categories simultaneously. We have a multi-label classification problem where typically represent the set of categories is often represented as a binary vector (indicating presence or absence of each category).
The methods that can be applied are like One-vs-Rest, Classifier Chains, or specialized neural network architectures that output multiple labels at once. The metrics would shift from simple accuracy to multi-label metrics like micro-F1, macro-F1, Hamming loss, for example.
    
        - How would you deploy this API on the cloud?

There are many ways to deploy a Dockerized FastAPI application to the cloud. A typical approach would be to use a container registry. You can build your Docker image locally and push it to a container registry such as Docker Hub, Amazon ECR, or Google Container Registry. After that, you choose a hosting service like AWS ECS or AWS EKS (Kubernetes).

Regarding networking and security, you need to expose the relevant port. Optionally, you can set up an API Gateway or a load balancer to manage traffic securely and at scale, and you may also integrate HTTPS certificates to secure the communication.

For CI/CD pipelines, you can automate builds and tests using tools such as GitHub Actions, GitLab CI, or Jenkins. Once the tests pass, you push the latest image to your container registry and update the cloud service accordingly. 

	    - If this model was deployed to categorize products without any supervision which metrics would you check to detect data drifting? When would you need to retrain?

When there is no continuous labeling or feedback, you must monitor changes in the data itself (inputs) and in the model’s output distribution. Relevant metrics and indicators include:

We have to check whether the distribution (mean, variance) of new embeddings deviates significantly from the training set’s embeddings. A large deviation suggests data drift, implying the model is seeing inputs that differ greatly from the training domain. Furthermore, monitor how frequently each category is predicted. If certain categories suddenly spike or disappear in prediction frequency, that may indicate the input data has changed and no longer matches the training data distribution.
It is usually a good pratice to retrain if you detect that the distribution of incoming data is significantly different from the training set of trained model.
