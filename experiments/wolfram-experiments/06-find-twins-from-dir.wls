#!/home/pmd/wolfram/bin/wolframscript
(* a lazy man's logger *)
log[begin_DateObject, s_String]:=Print[
	"After "<>ToString[Now-begin]<>": "<>s
];
begin = Now;
log[begin,"Here we go: "<>DateString[]]

peopleDir = FileNameJoin@{$HomeDirectory,"itusus"};
dataDir = "/data/daner/faces";
featuresFile = "/data/daner/all-features-faces.mx";
publishDir = FileNameJoin[{$HomeDirectory,"html","daner", "faces"}]

If[Not@DirectoryQ[publishDir],CreateDirectory[publishDir]]
If[Not@DirectoryQ[publishDir<>"/img"],CreateDirectory[publishDir<>"/img"]]

peopleFiles = FileNames["*.jpg", peopleDir, Infinity];
log[begin, "Loaded "<>ToString@Length@peopleFiles<>" portraits."]
imageFiles = FileNames["*.jpg",dataDir,Infinity];
log[begin, "Loaded "<>ToString@Length@imageFiles<>" paths to the faces which to search."]

(* load feature model featuresOfAllPhotos *)
Get@featuresFile
log[begin, "Feature model loaded with dimensions "<>ToString[Dimensions[featuresOfAllPhotos]]]
(* 
   Thread the list of features with the list of image paths 
   to get a list of rules for the nearest feature search.
   This should have been part of the data preparation!
*)
featureMap = Thread[Rule[featuresOfAllPhotos, imageFiles]];

featureExtractor=NetModel["ResNet-101 Trained on Augmented CASIA-WebFace Data"];
log[begin, "Model ResNet-101 Trained on Augmented CASIA-WebFace Data loaded."];

(* create the search function for nearest image. We want both the index of the nearest and the distance *)
findNearestPhoto = FeatureNearest[featuresOfAllPhotos -> "Index"];
log[begin, "Search function loaded."];

(* extract the features of the photos of the subjects *)
subjectFeatures = featureExtractor@Import[#]&/@peopleFiles;
log[begin, "Extracted features of "<>ToString@Length@peopleFiles<>" subject photos"];

findTwinsFast[originals_List] := Map[
  List[#, imageFiles[[First@findNearestPhoto@featureExtractor@Import@#]]]&, originals
];

(* find some twins and remember the time it took *)
log[begin, "Begin search for twins. This will take approximate "<>ToString[Quantity[7.5, "Seconds"]*Length@peopleFiles]]
{time, twins} = AbsoluteTiming[findTwinsFast[peopleFiles]];
log[begin, "Search complete in "<>ToString@time]

(* Create an image collage for each pair and export htose into the httpd area. *)
log[begin, "Create image collages and webpage"]
twinPairs = Export[
   FileNameJoin[{
      publishDir,
      "img", 
      FileNameTake[#[[1]]]
   }],
   ImageCollage[{
      Import[#[[1]]],
      Import[StringReplace[#[[2]],"faces"->"full_image"]]}]]&/@twins;


header = "<html><head/><body>";
tableHeader = "<table style=\"width:100%\">";
table = StringRiffle["<tr><td><img width=\"400px\" src=\"img/"<>FileNameTake@#<>"\"</img></td></tr>"&/@twinPairs, "\n"];
tableFooter = "</table>";
footer = "</body></html>";

htmlDoc = header<>tableHeader<>table<>tableFooter<>footer
DeleteFile[FileNameJoin[{publishDir,"index.html"}]]
WriteString[FileNameJoin[{publishDir,"index.html"}],htmlDoc]

log[begin, "Everything complete at "<>ToString@DateString[]]


