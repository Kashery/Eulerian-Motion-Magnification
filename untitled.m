clear all
close all
v = VideoReader('out.mkv');
currAxes = axes;
plotframe = [];
while hasFrame(v)
  frame = readFrame(v);
  F=  insertShape(frame,"line",[260 0 260 640],"ShapeColor","blue");
  
  plotframe = [plotframe  frame(:,260,:)];
  currAxes.Visible = 'off';
end
imshow(F);
out = plotframe;

v = VideoReader('in.mkv');
currAxes = axes;
plotframe = [];
while hasFrame(v)
  frame = readFrame(v);
  F=  insertShape(frame,"line",[450 580 450 580],"ShapeColor","blue");
  plotframe = [plotframe  frame(:,260,:)];
  currAxes.Visible = 'off';
end
figure
in = plotframe;
subplot(1,2,1);
imshow(in);
subplot(1,2,2);
imshow(out);