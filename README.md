# Wake-Cylinder Problem with Siren Model 

In this project, we tried to use SIREN model for solving cylinder wake simulation problem. By this we expect more smooth interpolating possible.

Important hyper-parameter of the SIREN model is the frequency omega, hence various frequencies have been tried. 

Authors suggest omega=30 works best for image fitting, omega=300 for audio fitting. However omega=3 seems to work best for this task.

Also, although normalized coordinates is used in SIREN, it seems to work even without it in this application.
