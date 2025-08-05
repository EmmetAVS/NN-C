"""
Helper file to transform your own images into a csv the model can use

Run ./example_digit_recognition with the filename(s) as params
ex: "./example_digit_recognition data/testing.csv data/mnist_train.csv"
"""

from PIL import Image
import numpy as np
import csv
import os

def generate_row(filename, label):
    image = Image.open(filename).convert('RGB').resize((28, 28))
    pixels = image.load()
    width, height = image.size
    
    img_data = np.zeros(height * width)
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            img_data[height * y + x] = int(((3 * 255) - (r + g + b))/3)
            
    return [label] + img_data.astype(int).tolist()
    
def generate_csv(images, labels):
    
    col_names = ["label"]
    for i in range(28 * 28):
        col_names.append(f"pixel{i}")
    
    rows = []
    for image, label in zip(images, labels):
        rows.append(generate_row(image, label))
        
    return (col_names, rows)

def write_csv(col_names, rows, filename):
    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        
        writer.writerow(col_names)
        writer.writerows(rows)
        
def main():
    csv = generate_csv(["data/image.png", "data/image2.png"], [7, 6])
    write_csv(csv[0], csv[1], "data/testing.csv")
    
if __name__ == "__main__":
    main()