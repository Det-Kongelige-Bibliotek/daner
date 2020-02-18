#!/home/pmd/wolfram/bin/wolframscript
begin = Now;

peopleDir = $ScriptCommandLine[[2]];
(*peopleDir = FileNameJoin@{$HomeDirectory,"projekter","daner","people-test"}*)

publishDir = FileNameJoin[{$HomeDirectory,"html","daner"}]
dataDir = "/data/daner/";

peopleFiles = FileNames["*.jpg", peopleDir, Infinity];
imageFiles = FileNames["*.jpg",FileNameJoin@{dataDir,"full_image"},Infinity];

Print["Loaded "<>ToString@Length@peopleFiles<>" portraits."]
Print["Elapsed time: "<>ToString[Now-begin]];


(* load feature model featuresOfAllPhotos *)
Get@FileNameJoin[{dataDir, "all-features.mx"}]
Print["Feature model loaded "]
Print["Elapsed time: "<>ToString[Now-begin]];
(* 
   Thread the list of features with the list of image paths 
   to get a list of rules for the nearest feature search.
   This should have been part of the data preparation!
*)
featureMap = Thread[Rule[featuresOfAllPhotos, imageFiles]];

featureExtractor=NetModel["ResNet-101 Trained on Augmented CASIA-WebFace Data"];

(* Reduce the amount of people for which to find a dobbelgänger *)
(*someKBPeople = RandomChoice[kbPeopleFiles,8];*)

(* create a function for finding pseudo twins for a list of portraits *)
findTwins[featureMap,originals_List] := Map[
  List[
     #, 
     First@FeatureNearest[
	featureMap, 
	featureExtractor@Import[#]
     ]
  ]&,
  originals
];

(* find some twins and remember the time it took *)
Print["Elapsed time: "<>ToString[Now-begin]];
Print["Begin search for twins. This will take approximate "<>ToString[Quantity[7.5, "Seconds"]*Length@peopleFiles]]
{time, twins} = AbsoluteTiming[findTwins[featureMap, peopleFiles]];
Print["Search complete in "<>ToString@time]
Print["Elapsed time: "<>ToString[Now-begin]];

(* Create an image collage for each pair and export htose into the httpd area. *)
Print["Create image collages"]
twinPairs = Export[
   FileNameJoin[{
      publishDir,
      "img", 
      FileNameTake[#[[1]]]
   }],
   ImageCollage[{
      Import[#[[1]]],
      Import[#[[2]]]}]]&/@twins;


header = "<html><head/><body>";
tableHeader = "<table style=\"width:100%\">";
table = StringRiffle["<tr><td><img width=\"400px\" src=\"img/"<>FileNameTake@#<>"\"</img></td></tr>"&/@twinPairs, "\n"];
tableFooter = "</table>";
footer = "</body></html>";

htmlDoc = header<>tableHeader<>table<>tableFooter<>footer
DeleteFile[FileNameJoin[{publishDir,"index.html"}]]
WriteString[FileNameJoin[{publishDir,"index.html"}],htmlDoc]

Print["Everything complete"]
Print["Elapsed time: "<>ToString[Now-begin]];

