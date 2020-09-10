---
title: NLTK Python App on Heroku
tags: nlp, machine learning
layout: post
date: 2019-12-01 01:30:00 -0400
comments: true
category: blog
description:
lang: en

---


# How to Run Your NLTK Pythonic App on Heroku

If you're having issues running your NLTK-based Pythonic app on Heroku, this guide is for you.



## Context

For my final project at GA, I was creating an [NLTK-based keyword assessment tool](https://github.com/duryan00/point_forward) that parses user-provided job descriptions and returns the relevant keywords that the user should prioritize on their resume.
* key package: NLTK

Additionally, I have a companion Word Cloud component that shares the same text preprocessing function as the above. It involves searching/scraping job descriptions for a user-input job title, passing that corpus through the preprocessing function before it is fed to the WordCloud generator.
* key packages: Selenium with Chromedriver (for webscraping)

## Problem

If you were to push the package as is to Heroku, the app will not run. That's because it is missing some core Python dependencies.

## Solution

To get around this, we need to install buildpacks to the Heroku app first!

The disadvantage to this approach is that buildpacks can take up a lot of space. Bear in mind that free Heroku only provides you with 500MB to play with.

Thus I would suggest that you avoid using buildpacks if you can help it, especially if your app may be computationally/memory intensive.

----

### NLTK Buildpack

**Step 1**:

Before you start, you ought to have had the new app created on Heroku first.

Once you have done so, navigate to "Settings" and scroll down to `Buildpacks`. Click on "Add buildpack" and paste the following:

> `heroku/python`

Click "Save changes".

**Step 2:**

> `touch runtime.txt`

> `echo "python-3.7.3" >> runtime.txt`

What this does is create a file that specifies the Python version you want to run on your Heroku app. Feel free to change `3.7.3` to whichever Python version you desire.

**Step 3**:

Next you'll need to create a `nltk.txt` file and populate it. You can do it the old-fashioned way, or create and populate it from the terminal (make sure your terminal is pointing to the app's directory).

> `touch nltk.txt`

> `echo "wordnet" >> nltk.txt`

In my case I require the `wordnet` and `stopwords` packages from NLTK to be installed, so I'll do this instead:

> `touch nltk.txt`

> `echo -e "wordnet \\nstopwords" >> nltk.txt`

---

### Selenium/Chromedriver Buildpack

Just as in `Step 1` in the NLTK section above, paste and add the following to your Buildpacks.

> `https://github.com/heroku/heroku-buildpack-google-chrome`

> `https://github.com/heroku/heroku-buildpack-chromedriver`

---

### Bonus: Auto `requirements.txt` Generator

To ensure you get the correct versions of packages used in your app, and just in case you forgot to run a **venv**, [pipreqs](https://github.com/bndr/pipreqs) is your friend.

Install:
> `pip install pipreqs`

Usage:

From the terminal, change directory to point to your app folder where your **app.py** is located. Then just key in the following:

> `pipreqs`

You should see a new `requirements.txt` file generated automatically!
