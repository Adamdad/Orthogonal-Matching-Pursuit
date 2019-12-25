% Report class created by Dan Horne
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ucsdreport}[2019/01/12 Report LaTeX class]
\LoadClass[]{article}
\usepackage{subfigure}
% PACKAGE DECLARATIONS
% general page layout and fonts
\usepackage[margin= 1in]{geometry}
\usepackage[utf8]{inputenc}
\usepackage[none]{hyphenat}
\usepackage{parskip}
\usepackage{multicol}
% standard packages
\usepackage{enumerate}
\usepackage{scrextend}
\usepackage{fancyhdr}
% math packages
\usepackage{centernot}
\usepackage{amsthm, amssymb, amsmath,verbatim}
\usepackage{mathtools}
% coding and colors
\usepackage{xifthen}
\usepackage{ifthen}
% color box
\usepackage[colorlinks=true,linkcolor=blue]{hyperref}
\usepackage[most,listings]{tcolorbox}
\usepackage{lmodern}
% graphs and pictures
\usepackage{tikz}
\usepackage{mathrsfs}
\usetikzlibrary{math}
\usetikzlibrary{backgrounds}
\usetikzlibrary{patterns,calc}
\usepackage{graphicx, subcaption}
\usepackage{csvsimple,booktabs}
\usepackage{filecontents}

\usepackage{algorithm}
\usepackage[noend]{algpseudocode}

% ARRAY COMMANDS
\usepackage{array}
\newcommand{\PreserveBackslash}[1]{\let\temp=\\#1\let\\=\temp}
\newcolumntype{C}[1]{>{\PreserveBackslash\centering}p{#1}}
\newcolumntype{R}[1]{>{\PreserveBackslash\raggedleft}p{#1}}
\newcolumntype{L}[1]{>{\PreserveBackslash\raggedright}p{#1}}

% HEADER
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{.5pt}

\fancyfoot[L]{\Headauthor}
\fancyfoot[R]{\thetitle \quad\textbf{\thepage}
}
\fancypagestyle{plain}{%
\fancyhf{}
\renewcommand{\footrulewidth}{.5pt}
\fancyfoot[L]{\author}
\fancyfoot[R]{\thetitle \quad\textbf{\thepage}
} 
}
% DEFINE ENVIRONMENTS

