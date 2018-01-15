# -*- coding: utf-8 -*-
from subprocess import Popen, PIPE, STDOUT
import os
from PIL import Image
import eyed3

from sliceSpectrogram import createSlicesFromSpectrograms
from audioFilesTools import isMono, getGenre
from config import rawDataPath
from config import spectrogramsPath
from config import pixelPerSecond

#Tweakable parameters
desiredSize = 128

#Define
currentPath = os.path.dirname(os.path.realpath(__file__)) 

#Remove logs
eyed3.log.setLevel("ERROR")

#Create spectrogram from mp3 files
def createSpectrogram(filename,newFilename, genre):
	#Create temporary mono track if needed
	file_path = rawDataPath + '/' + genre + '/' + filename
	if isMono(file_path):
		command = "cp '{}' 'tmp/{}.mp3'".format(file_path,newFilename)
	else:
		command = "sox '{}' 'tmp/{}.mp3' remix 1,2".format(file_path,newFilename)
	p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=currentPath)
	print(command)
	output, errors = p.communicate()
	if errors:
		print(errors)

	#Create spectrogram
	filename.replace(".mp3","")
	command = "sox 'tmp/{}.mp3' -n spectrogram -Y 200 -X {} -m -r -o '{}.png'".format(newFilename,pixelPerSecond,spectrogramsPath+newFilename)
	p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=currentPath)
	# print(command)
	output, errors = p.communicate()
	if errors:
		print(errors)

	#Remove tmp mono track
	try:
		os.remove("tmp/{}.mp3".format(newFilename))
	except Exception as e:
		print(e)

#Creates .png whole spectrograms from mp3 files
def createSpectrogramsFromAudio():

	for genre_dir in os.listdir(rawDataPath):

		if '.DS_Store' == genre_dir:
			continue

		_rawDataPath = rawDataPath +'/'+ genre_dir + '/'
		_genre_type = genre_dir
		print(_rawDataPath)
		print(genre_dir)
		# continue
		genresID = dict()

		files = os.listdir(_rawDataPath)
		files = [file for file in files if file.endswith(".mp3")]
		nbFiles = len(files)

		#Create path if not existing
		if not os.path.exists(os.path.dirname(spectrogramsPath)):
			try:
				os.makedirs(os.path.dirname(spectrogramsPath))
			except OSError as exc: # Guard against race condition
				if exc.errno != errno.EEXIST:
					raise

		#Rename files according to genre
		for index,filename in enumerate(files):
			
			print("Creating spectrogram for file {}/{}...".format(index+1,nbFiles))
			# fileGenre = getGenre(_rawDataPath+filename)
			fileGenre = _genre_type
			genresID[fileGenre] = genresID[fileGenre] + 1 if fileGenre in genresID else 1
			fileID = genresID[fileGenre]
			newFilename = fileGenre+"_"+str(fileID)
			createSpectrogram(filename,newFilename, fileGenre)

#Whole pipeline .mp3 -> .png slices
def createSlicesFromAudio():
	print("Creating spectrograms...")
	createSpectrogramsFromAudio()
	print("Spectrograms created!")

	print("Creating slices...")
	createSlicesFromSpectrograms(desiredSize)
	print("Slices created!")