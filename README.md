# Seam Carving for Content-Aware Image Resizing
This project replicates the findings of the original paper "Seam Carving for Content-Aware Image Resizing". Seam-carving is a content-aware image resizing method that adjusts the image's height or width by one pixel at a time.

## Introduction
Seam carving provides content-aware image scaling, allowing for image resizing without the loss of significant content due to cropping or scaling.

## Dependencies
* numpy
* OpenCV (cv2)
* sys
* imutils
* numba

## Key Functions
1. __setsize(input, resultheight, resultwidth):__ Resizes an input image based on the specified dimensions.
2. __getseam(image, energy):__ Retrieves the seam for a given image based on its energy.
3. __addseam(seam, image):__ Inserts a seam into the specified image.
4. __delseam(image, seam):__ Removes a seam from the provided image.
5. __newseam(left, present):__ Calculates or updates seam coordinates.
6. __getmap(image):__ Computes the energy map for the input image.

## How to Run
1. Place the seam.py file and images in the same directory (avoid nested folders for images).
2. Open the command line and navigate to the directory containing the seam.py file and images.
3. Execute the Python file. Example images are provided with the code for testing.
4. Note: Only use .png images as input, as this program saves outputs in .png format.

## Warnings and Shortcomings
During execution, multiple deprecation warnings may arise due to the use of the numba library. Additionally, errors might be observed when trying to save an image with the minimum seam line.

## Results and Demonstrations
The project showcases various results, including images with their energy maps, images with modified dimensions, and distorted results due to the seam carving process.

## Related Work and Citations
The seam carving algorithm was introduced by Shai Avidan and Ariel Shamir in 2007. Many improvements to the algorithm have been introduced since its inception.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.
