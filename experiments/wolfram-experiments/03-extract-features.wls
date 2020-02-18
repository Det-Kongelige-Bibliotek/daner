#!/home/pmd/wolfram/bin/wolframscript

dataDir = "/data/daner/full_image";
imageFiles = FileNames["*.jpg",dataDir,Infinity];

(*imageFiles = RandomChoice[imageFiles,200];*)

featureExtractor=NetModel["ResNet-101 Trained on Augmented CASIA-WebFace Data"]

Print["Begin: "<>DateString@Now]
featuresOfAllPhotos = featureExtractor@Import@#&/@imageFiles;
Print["End: "<>DateString@Now]

(*
Dimensions@featuresOfAllPhotos
Short@featuresOfAllPhotos
MemoryInUse[]
MaxMemoryUsed[]
*)


DumpSave["all-features.mx",featuresOfAllPhotos];
Print["Done extracting 2048 features from "<>ToString@Length@imageFiles<>" images"]

