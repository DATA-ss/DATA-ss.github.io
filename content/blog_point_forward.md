---
title: Point Forward: The Job Description Hacker
tags: machine learning, natural language processing
layout: post
date: 2019-11-12 03:31:00 -0400
comments: true
category: blog
description: 
lang: en

---

As the DSI course winds to a close, I've set out to produce a NLP-based model that should "hack" a job description and returns the top most words one should use in their job search efforts (in this case me!).

There would be two parts to this project before it is uploaded to Heroku:

1) Scraping

2) Modeling

### Scraping

To gather the necessary data, I've decided to scrape the job directories at Indeed and Linkedin. The packages necessary to achieve this are `gazpacho`, `BeautifulSoup` and `Selenium`.

A challenge I faced in scraping them is that these websites do not have clearly defined templates for the postings; the `<div>` tags are unorganized/unnamed and as such its difficult to remove the fluff text that usually accompanies the job description and qualifications that are being sought.

One workaround is to scrape only the `<li>` list tags instead, as they usually contain the proverbial 'meat' that we want to have on our resumes.

To that end, the repo will have two versions of the scraping script: one to scrape everything under a specified `<div>` tag, and the other to scrape only `<li>` tags.

### Modeling

To model this adequately, `spaCy` is the most suitable package to support this project.

The idea is to have a front-end that accepts a user-input (via copy-paste) of a job description they found online, and the model would parse it for the most relevant texts that the user should use in their resume to target that specific job.

## *STAY TUNED!!*
