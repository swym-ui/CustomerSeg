from flask import Flask, render_template, request
import pandas as pd
from sklearn.cluster import KMeans
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def assign_segment(row):
    income = row['Annual Income (k$)']
    spending = row['Spending Score (1-100)']
    if income > 70 and spending > 60:
        return 'Luxury Shopper'
    elif income > 70 and spending < 40:
        return 'Wealthy Saver'
    elif income < 40 and spending > 60:
        return 'Impulsive Buyer'
    else:
        return 'Budget Conscious'

def assign_offer(segment):
    return {
        'Luxury Shopper': 'Premium Products Only',
        'Wealthy Saver': 'Targeted Discounts',
        'Impulsive Buyer': 'Buy 2 Get 1 Free',
        'Budget Conscious': 'Budget Combos'
    }.get(segment, 'General Offer')

def assign_products(segment):
    return {
        'Luxury Shopper': 'Watches, Perfume, Jewelry',
        'Wealthy Saver': 'Laptops, Travel Packages',
        'Impulsive Buyer': 'Snacks, Headphones',
        'Budget Conscious': 'Groceries, Basic Clothing'
    }.get(segment, 'Various')

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            data = pd.read_csv(filepath)
            features = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

            kmeans = KMeans(n_clusters=3, random_state=0)
            data['Cluster'] = kmeans.fit_predict(features)

            data['Segment'] = data.apply(assign_segment, axis=1)
            data['Suggested Offer'] = data['Segment'].apply(assign_offer)
            data['Likely Products'] = data['Segment'].apply(assign_products)

            summary = data.groupby('Segment').agg({
                'CustomerID': 'count',
                'Annual Income (k$)': 'mean',
                'Spending Score (1-100)': 'mean'
            }).rename(columns={
                'CustomerID': 'Total Customers',
                'Annual Income (k$)': 'Avg Income',
                'Spending Score (1-100)': 'Avg Spending Score'
            }).reset_index()

            summary['Offer Type'] = summary['Segment'].apply(assign_offer)
            summary['Estimated Revenue'] = summary['Total Customers'] * summary['Avg Spending Score']

            return render_template("results.html", summary=summary.to_dict(orient="records"))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
