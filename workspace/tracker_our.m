
% error('Tracker not configured! Please edit the tracker_our.m file.'); % Remove this line after proper configuration

% The human readable label for the tracker, used to identify the tracker in reports
% If not set, it will be set to the same value as the identifier.
% It does not have to be unique, but it is best that it is.
tracker_label = ['our'];

% For Python implementations we have created a handy function that generates the appropritate
% command that will run the python executable and execute the given script that includes your
% tracker implementation.
%
% Please customize the line below by substituting the first argument with the name of the
% script of your tracker (not the .py file but just the name of the script) and also provide the
% path (or multiple paths) where the tracker sources % are found as the elements of the cell
% array (second argument).
addpath('/home/jaehyuk/anaconda2/lib')
tracker_command = generate_python_command('python_coord', {'/home/jaehyuk/code/github/vot-toolkit/tracker/examples/python/'});

tracker_interpreter = 'python';

% tracker_linkpath = {}; % A cell array of custom library directories used by the tracker executable (optional)
