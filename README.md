# Introduction
So, I had this discussion with a former collegue of mine, He was interested in using Machine learning and Artificial Intelligence to build a add-on to his existing SMSC platform. The add-on is supposed to work as a SMS firewall which could dynamically enforce rules and policies on SMSC to facilitate various purposes. 

## Features
Typically an SMS firewall shall have following properties in-built within it's scope to enable and serve it's purpose. Those are 
- **Spamming-Resistance** : A system which could block unsolicited SMS messages to a subscriber
- **Flooding-Resistance** : The system should be able to block large amount of messages to various destinations 
- **Spoofing-Resistance**:  System should be able to block using the SMS client to send manupilated messages. 

I agree, apart from above there are many other features that a typical SMSC firewall could support. but as of now Let's stop here and focus on the main focus of this module; 

## Purpose
Purpose of this module is to build the spamming-resistance capabilities using machine learning in a typical SMS enviornment. This is intended to be taken as a Proof-Of-Concept(POC) of explaining and demonstrating such capabilities to the end-user

## Background
To build a machine learning platform expressing the POC nature of above mentioned factuals, I have used the following
- SciKit-Learn
- Natural Language Toolkit
- Source Dataset from One of the Kaggle Competitions
- Bit of Mathematics and Lots of Trial and Error

## What is In It ?
The POC does the following tasks 
- Clean the input data 
- Prepare the labels
- Train and Test set of models ( Reason is to chose the best model out of possiblities) 
- Create a pipeline
- Save the pipeline
- Load/Test the pipeline for deployment

## Do you find it Interesting ?
Comments below..
