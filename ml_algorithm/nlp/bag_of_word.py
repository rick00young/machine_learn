from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
corpus = [
	'I come to china to travel',
	'this is a car polupar in china',
	'i love tea and apple',
	'the work is to write some papers in secience'
]

print(vectorizer.fit_transform(corpus))


from sklearn.feature_extraction.text import HashingVectorizer
vectorizer2 = HashingVectorizer(non_negative=6, norm=None)
print('HashingVectorizer')
print(vectorizer2.fit_transform(corpus))