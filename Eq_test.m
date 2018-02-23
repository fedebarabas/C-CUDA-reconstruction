pat_per = 20;
pat_start = 10;

j = 0:100;
y = j/pat_per + 1/2 - pat_start/pat_per;

plot(j, floor(y))