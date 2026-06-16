%% dynamics with delta term
h=0.00001;                                             
t = 0:h:50;

S=100;
p=0.2;
m1=30;
m2=30;

Q1=2*[-1/p,1/(2*p);1/(2*p),0];
b1=[(m1+S*p)/p,-1*m1/p]';
Q2=2*[0,1/(2*p);1/(2*p),-1/p];
b2=[-1*m2/p,m2/p]';

% Q1=[3 1;1 5];
% b1=[4 2]';
% Q2=[7 2;2 4];
% b2=[1 6]';



A0=[Q1(1,:);Q2(2,:)];
b0=[b1(1),b2(2)]';
x0=-1*inv(A0)*b0;

%c1 and c2
c1=-1*(Q1(1,1)+Q2(2,2))/Q1(1,2);
c2=det(A0)/(Q2(1,2)*Q1(2,2)-Q2(2,2)*Q1(1,2));

%functions
w=[1 -1*Q2(1,2)/Q2(2,2)]';
q1=-1*(b1(2)+Q1(2,:)*x0);
q2=Q1(2,:)*w;
q3=Q1(1,:)*w;
r2=0.5*w'*Q2*w;
r1=(b2+Q2*x0)'*w;


a=0.05;
k=-0.03;
w1=12877.75;
w2=11436.5;


e_delta=@(d) d*q1/(d*q2+q3);
J1=@(x) 0.5*x'*Q1*x+b1'*x-3000;
J2=@(x) 0.5*x'*Q2*x+b2'*x;

J2quad=@(e) r2*e^2+r1*e+J2(x0);
%J1=@(x) (x-5).^2;
%J2=@(x) (x-10).^2;

epsilon=-0.001;
ep2=-0.05;
J2ref=1000;
t1=.5;
t2=3.9;
prices=@(t,u,d) u+a*[sin(w1*t), sin(w2*t)+(d)*sin(w1*t)]';
udot=@(time,u) [(-2*k/a)*J1(prices(time,u(1:2),(t2/t1)*u(4)-((t2/t1)-1)*u(3)))*sin(w1*time),(-2*k/a)*J2(prices(time,u(1:2),(t2/t1)*u(4)-((t2/t1)-1)*u(3)))*sin(w2*time),-(1/t1)*u(3)+(1/t1)*u(4),epsilon*(J2(prices(time,u(1:2),(t2/t1)*u(4)-((t2/t1)-1)*u(3)))-J2ref)]';
%udot=@(time,u) [(-2*k/a)*J1(u(1:2)+a*[sin(w1*time),sin(w2*time)+u(3)*sin(w1*time)]')*sin(w1*time),(-2*k/a)*J2(u(1:2)+a*[sin(w1*time),sin(w2*time)+u(3)*sin(w1*time)]')*sin(w2*time),epsilon*(J2(u(1:2)+a*[sin(w1*time),sin(w2*time)+u(3)*sin(w1*time)]')-J2ref)]';
u = zeros(length(t),4);
u(1,:)=[50,110/3,0,0];
%u(1,:)=[0,0];
J=zeros(length(t)-1,2);

for i=1:(length(t)-1) 
%     u(i,3)=0; 
%     u(i,4)=0;
    k_1 = udot(t(i),u(i,:)');
    k_2 = udot(t(i)+0.5*h,u(i,:)'+0.5*h*k_1);
    k_3 = udot((t(i)+0.5*h),(u(i,:)'+0.5*h*k_2));
    k_4 = udot((t(i)+h),(u(i,:)'+k_3*h));
    u(i+1,:) = u(i,:)' + (1/6)*(k_1+2*k_2+2*k_3+k_4)*h;  % main equation
    %delta=delta+ -1*epsilon*(J2(u(i,:)')-J2ref)*h;
    J(i,1)=J1(u(i,1:2)');
    J(i,2)=J2(u(i,1:2)');
  
end
%%
%xd=-1*inv(A0+u(333334,3)*[Q1(2,:);0 0])*(b0+u(333334,3)*[b1(2);0]);
figure
plot(t,u(:,1),'LineWidth',2)
hold on
plot(t,u(:,2),'LineWidth',2)
%ylim([34 75]);
hold on 

xd=-1*inv(A0+.7928*[Q1(2,:);0 0])*(b0+.7928*[b1(2);0]);
for i=1:2
    plot(t,ones(1,length(t))*xd(i),'--','color','black','LineWidth',2)
    hold on
end
xlabel('Time (s)')
ax=gca;
ax.FontSize = 15;
labelj=ylabel('$x_i$ ','Interpreter','latex','Rotation',0,'FontSize',25);
labelj.Position(1) = -3;
legend('$x_1$','$x_2$','DNE','Interpreter','latex','FontSize',25)
%ylim([30 73])
%title('Duopoly with Compensator')
print('duo_comp_x','-depsc')
%%
clf
plot(t, u(:,4))
%%
figure
lenJ=length(t)-1;

plot(t(1:lenJ),J(:,1),'LineWidth',2)
hold on
plot(t(1:lenJ),J(:,2),'LineWidth',2)
hold on
plot(t(1:lenJ),ones(1,lenJ)*1000,'--','color','black','LineWidth',2)
legend('$J_1$', '$J_2$', '$J_2^{ref}$','Position',[.5 .23 .25 .2],'FontSize',25,'Interpreter','latex')
%ylim([0 1400])
%title('Duopoly with Compensator')
ax=gca;
ax.FontSize = 15;
xlabel('Time (s)')
ylabel('Profit')
print('duo_comp_j','-depsc')
%%
J_nodec=J;
%% dynamics with delta term
clf
h=0.0001;                                             
t = 0:h:100;

S=100;
p=0.2;
m1=30;
m2=30;

Q1=2*[-1/p,1/(2*p);1/(2*p),0];
b1=[(m1+S*p)/p,-1*m1/p]';
Q2=2*[0,1/(2*p);1/(2*p),-1/p];
b2=[-1*m2/p,m2/p]';

% Q1=[3 1;1 5];
% b1=[4 2]';
% Q2=[7 2;2 4];
% b2=[1 6]';



A0=[Q1(1,:);Q2(2,:)];
b0=[b1(1),b2(2)]';
x0=-1*inv(A0)*b0;

%c1 and c2
c1=-1*(Q1(1,1)+Q2(2,2))/Q1(1,2);
c2=det(A0)/(Q2(1,2)*Q1(2,2)-Q2(2,2)*Q1(1,2));

%functions
w=[1 -1*Q2(1,2)/Q2(2,2)]';
q1=-1*(b1(2)+Q1(2,:)*x0);
q2=Q1(2,:)*w;
q3=Q1(1,:)*w;
r2=0.5*w'*Q2*w;
r1=(b2+Q2*x0)'*w;



a=0.05;
k=-0.03;
w1=7877.75;
w2=73436.5;


e_delta=@(d) d*q1/(d*q2+q3);
J1=@(x) 0.5*x'*Q1*x+b1'*x-3000;
J2=@(x) 0.5*x'*Q2*x+b2'*x;

J2quad=@(e) r2*e^2+r1*e+J2(x0);
%J1=@(x) (x-5).^2;
%J2=@(x) (x-10).^2;

epsilon=-0.001;
ep2=-0.05;
J2ref=1000;
prices=@(t,u,d) u+a*[sin(w1*t), sin(w2*t)+(d)*sin(w1*t)]';
udot=@(time,u) [(-2*k/a)*J1(prices(time,u(1:2),u(3)))*sin(w1*time),(-2*k/a)*J2(prices(time,u(1:2),u(3)))*sin(w2*time),epsilon*(J2(prices(time,u(1:2),u(3)))-J2ref)]';
%udot=@(time,u) [(-2*k/a)*J1(u(1:2)+a*[sin(w1*time),sin(w2*time)+u(3)*sin(w1*time)]')*sin(w1*time),(-2*k/a)*J2(u(1:2)+a*[sin(w1*time),sin(w2*time)+u(3)*sin(w1*time)]')*sin(w2*time),epsilon*(J2(u(1:2)+a*[sin(w1*time),sin(w2*time)+u(3)*sin(w1*time)]')-J2ref)]';
u = zeros(length(t),3);
u(1,:)=[50,110/3,0];
%u(1,:)=[0,0];
J=zeros(length(t)-1,2);

for i=1:(length(t)-1) 
    %u(i,3)=0; 
    k_1 = udot(t(i),u(i,:)');
    k_2 = udot(t(i)+0.5*h,u(i,:)'+0.5*h*k_1);
    k_3 = udot((t(i)+0.5*h),(u(i,:)'+0.5*h*k_2));
    k_4 = udot((t(i)+h),(u(i,:)'+k_3*h));
    u(i+1,:) = u(i,:)' + (1/6)*(k_1+2*k_2+2*k_3+k_4)*h;  % main equation
    %delta=delta+ -1*epsilon*(J2(u(i,:)')-J2ref)*h;
    J(i,1)=J1(u(i,1:2)');
    J(i,2)=J2(u(i,1:2)');
  
end

%xd=-1*inv(A0+u(333334,3)*[Q1(2,:);0 0])*(b0+u(333334,3)*[b1(2);0]);
plot(t,u(:,1))
hold on
plot(t,u(:,2))


% for i=1:2
%     plot(t,ones(1,length(t))*xd(i),'--','color','black','LineWidth',2)
%     hold on
% end
xlabel('time (s)')
ylabel('$x_i$ ','Interpreter','latex')
% legend('$x_1$','$x_2$','DNE','Interpreter','latex')
ax=gca;
ax.FontSize = 15;
title('Price convergence with deception')
print('duo_nodec','-depsc')
%%
clf
step=5000;
plot_nodec(1)=plot(t(1:step:end),u(1:step:end,1),':','Linewidth',3,'Color','b');
hold on
plot_nodec(2)=plot(t(1:step:end),u(1:step:end,2),':','Linewidth',3,'Color','r');
hold on
% ah2 = axes('position',get(gca,'position'),'visible','off');
% legend(plot_nodec(:),'x_1', 'x_2','Position',[.7 .7 .1 .1])
plot_dec(1)=plot(t,ud(:,1),'Linewidth',3,'Color',"b");
hold on
plot_dec(2)=plot(t(1:end),ud(1:end,2),'Linewidth',3,'Color',"r");

hold on
plot_x0(1)=plot(t, ones(1, length(t))*x0(1),'--','Color','k','LineWidth',2);
hold on
plot_x0(2)=plot(t, ones(1, length(t))*x0(2),'--','Color','k','LineWidth',2);
hold on
plot_xd(1)=plot(t, ones(1, length(t))*xd(1),'Color','k','LineWidth',2);
hold on
plot_xd(2)=plot(t, ones(1, length(t))*xd(2),'Color','k','LineWidth',2);
ax=gca;
ax.FontSize = 15;
ylabel('Action')
xlabel('Time (s)')
%ylim([35 60])
print('decxx','-depsc')
ah1 = axes('position',get(gca,'position'),'visible','off');
legend(ah1,plot_nodec(:),'$x_1$', '$x_2$','Position',[.7 .73 .17 .17],'Interpreter','latex','FontSize',20)

ah2 = axes('position',get(gca,'position'),'visible','off');
legend(ah2,plot_dec(:),'$x_1$ with deception', '$x_2$ with deception','Position',[.6 .45 .2 .2],'Interpreter','latex','FontSize',20)

ah3 = axes('position',get(gca,'position'),'visible','off');
legend(ah3,plot_x0(1),'$x^*$','Position',[.7 .73 .17 .17],'Interpreter','latex','FontSize',20)

ah4 = axes('position',get(gca,'position'),'visible','off');
legend(ah4,plot_xd(1),'$x_\delta$','Position',[.7 .73 .17 .17],'Interpreter','latex','FontSize',20)

% plot(t,ones(1,length(t))*jref(1),'--','color','black','LineWidth',2)
% hold on
% plot(t,ones(1,length(t))*jref(2),'-','color','black','LineWidth',2)
% hold on
ax=gca;
ax.FontSize = 15;
% title('Duopoly prices')
% ylabel('x')
% xlabel('time (s)')
%print('decxx','-depsc')
%%
clf
lent=length(t)-1;
step=10000;
jp(1)=plot(t(1:step:lent),J(1:step:end,1),':','Linewidth',3,'Color','b');
hold on
jp(2)=plot(t(1:step:lent),J(1:step:end,2),':','Linewidth',3,'Color','r');
hold on
jdp(1)=plot(t(1:lent),jd(:,1),'Linewidth',2,'Color',"b");
hold on
jdp(2)=plot(t(1:lent),jd(:,2),'Linewidth',2,'Color',"r");
hold on
jdp(3)=plot(t,ones(1,length(t))*1000,'--','color','black','LineWidth',2);
ax=gca;
ax.FontSize = 15;
xlabel('Time (s)')
labelj=ylabel('$J_i$','Interpreter','latex','Rotation',0,'FontSize',20);
%labelj.Position(1) = -14;
ylim([0 1700])
%legend('$J_1$','$J_2$','$J_1$ with deception', '$J_2$ with deception', '$J_2^{ref}$','Interpreter','latex','FontSize',13,'Position',[.55 .65 .2 .2])
ah1 = axes('position',get(gca,'position'),'visible','off');
legend(ah1,jp(:),'$J_1$','$J_2$','Position',[.7 .73 .17 .17],'Interpreter','latex','FontSize',22)

ah2 = axes('position',get(gca,'position'),'visible','off');
legend(ah2,jdp(:),'$J_1$ with deception', '$J_2$ with deception', '$J_2^{ref}$','Position',[.6 .45 .2 .2],'Interpreter','latex','FontSize',18)

% axes('Position',[.55 .28 .3 .18])
% box on
% plot(t, u(:,3),'Linewidth',2,'Color','m')
% ylabel('\delta', 'FontWeight','bold')
% 
% title('Dynamics of \delta')
print('decj','-depsc')
%%
clf
%plot(t, (1-t2/t1)*u(:,3)+t2/t1*u(:,4),'Linewidth',2,'Color','k')
plot(t, u(:,3),'Linewidth',2,'Color','r')
ax=gca;
ax.FontSize = 15;
labeld=ylabel('$\delta$','Interpreter','latex','Rotation',0,'FontSize',20);
% labeld.Position(1) = -10;
% labeld.Position(2) = .84;
%title('Dynamics of $\delta$','Interpreter','latex')
hold on
ssd=plot(t, ones(length(t),1)*0.7928,'--','Linewidth',2,'Color','k');
legend(ssd, "$\delta^*=0.7928$", 'Interpreter','latex','FontSize',25,'Position',[.5 .67 .3 .1])
xlabel("Time (s)")
ylim([0 2])
print('deltaduo','-depsc')
%% players 1 and 2 deceive each other
clf
h=0.0001;                                             
t = 0:h:1000;

S=100;
p=0.2;
m1=30;
m2=30;

Q1=2*[-1/p,1/(2*p);1/(2*p),0];
b1=[(m1+S*p)/p,-1*m1/p]';
Q2=2*[0,1/(2*p);1/(2*p),-1/p];
b2=[-1*m2/p,m2/p]';

% Q1=[3 1;1 5];
% b1=[4 2]';
% Q2=[7 2;2 4];
% b2=[1 6]';



A0=[Q1(1,:);Q2(2,:)];
b0=[b1(1),b2(2)]';
x0=-1*inv(A0)*b0;

%c1 and c2
c1=-1*(Q1(1,1)+Q2(2,2))/Q1(1,2);
c2=det(A0)/(Q2(1,2)*Q1(2,2)-Q2(2,2)*Q1(1,2));

%functions
w=[1 -1*Q2(1,2)/Q2(2,2)]';
q1=-1*(b1(2)+Q1(2,:)*x0);
q2=Q1(2,:)*w;
q3=Q1(1,:)*w;
r2=0.5*w'*Q2*w;
r1=(b2+Q2*x0)'*w;



a=0.05;
k=-0.03;
w1=11877.75;
w2=12436.5;


e_delta=@(d) d*q1/(d*q2+q3);
J1=@(x) 0.5*x'*Q1*x+b1'*x-3000;
J2=@(x) 0.5*x'*Q2*x+b2'*x;

J2quad=@(e) r2*e^2+r1*e+J2(x0);
%J1=@(x) (x-5).^2;
%J2=@(x) (x-10).^2;

epsilon1=-0.001;
epsilon2=-0.0005;
J2ref=1800;
J1ref=1200;
prices=@(t,u,d1, d2) u+a*[sin(w1*t)+d1*sin(w2*t), sin(w2*t)+(d2)*sin(w1*t)]';
udot=@(time,u) [(-2*k/a)*J1(prices(time,u(1:2),u(3), u(4)))*sin(w1*time),(-2*k/a)*J2(prices(time,u(1:2),u(3), u(4)))*sin(w2*time),epsilon1*(J1(prices(time,u(1:2),u(3), u(4)))-J1ref),epsilon2*(J2(prices(time,u(1:2),u(3),u(4)))-J2ref)]';
%udot=@(time,u) [(-2*k/a)*J1(u(1:2)+a*[sin(w1*time),sin(w2*time)+u(3)*sin(w1*time)]')*sin(w1*time),(-2*k/a)*J2(u(1:2)+a*[sin(w1*time),sin(w2*time)+u(3)*sin(w1*time)]')*sin(w2*time),epsilon*(J2(u(1:2)+a*[sin(w1*time),sin(w2*time)+u(3)*sin(w1*time)]')-J2ref)]';
u = zeros(length(t),4);
u(1,:)=[50,100/3,0,0];
%u(1,:)=[0,0];
J=zeros(length(t)-1,2);

for i=1:(length(t)-1) 
    %u(i,4)=0; 
    k_1 = udot(t(i),u(i,:)');
    k_2 = udot(t(i)+0.5*h,u(i,:)'+0.5*h*k_1);
    k_3 = udot((t(i)+0.5*h),(u(i,:)'+0.5*h*k_2));
    k_4 = udot((t(i)+h),(u(i,:)'+k_3*h));
    u(i+1,:) = u(i,:)' + (1/6)*(k_1+2*k_2+2*k_3+k_4)*h;  % main equation
    %delta=delta+ -1*epsilon*(J2(u(i,:)')-J2ref)*h;
    J(i,1)=J1(u(i,1:2)');
    J(i,2)=J2(u(i,1:2)');
  
end

xd=-1*inv(A0+u(333334,3)*[Q1(2,:);0 0])*(b0+u(333334,3)*[b1(2);0]);
plot(t,u(:,1),'LineWidth',2)
hold on
plot(t,u(:,2),'LineWidth',2)


% for i=1:2
%     plot(t,ones(1,length(t))*xd(i),'--','color','black','LineWidth',2)
%     hold on
% end
xlabel('time (s)')
ylabel('$x_i$ ','Interpreter','latex')
% legend('$x_1$','$x_2$','DNE','Interpreter','latex')
ax=gca;
ax.FontSize = 15;
title('Price convergence with deception')
print('duo_nodec','-depsc')
%%
clf
jref=[J1ref J2ref];
lenJ=length(t)-1;
plot(t(1:lenJ),J(:,1),'LineWidth',2)
hold on
plot(t(1:lenJ),J(:,2),'LineWidth',2)
hold on
% for i=1:2
%     plot(t,ones(1,length(t))*jref(i),'--','color','black','LineWidth',2)
%     hold on
% end
plot(t,ones(1,length(t))*jref(1),'-','color','black','LineWidth',2)
hold on
plot(t,ones(1,length(t))*jref(2),'--','color','black','LineWidth',2)
hold on
lgd=legend('$J_1$', '$J_2$', '$J_1^{ref}$', '$J_2^{ref}$','Interpreter','latex','Position',[.59 .62 .22 .2]);
lgd.FontSize = 20;
ylabel('Profit')
xlabel('Time (s)')
%ylim([0 4000])
ax=gca;
ax.FontSize = 15;
%title('Mutual Deception')
print('mutualdec','-depsc')
%%
clf
subplot(2,1,1)
plot(t, u(:,3),'Linewidth',2,'Color','k')
ax=gca;
ax.FontSize = 15;
% labeld=ylabel('$\delta$','Interpreter','latex','Rotation',0,'FontSize',35);
% labeld.Position(1) = -90;
% labeld.Position(2) = 0;
%title('Dynamics of $\delta$','Interpreter','latex')
hold on
plot(t, u(:,4),'Linewidth',2,'Color','r')
% ssd=plot(t, ones(length(t),1)*0.7928,'--','Linewidth',2);
legend("$\delta_1$", "$\delta_2$", 'Interpreter','latex','FontSize',25,'Position',[.5 .67 .2 .1],'FontSize',18)

subplot(2,1,2)
t2=0:h:50;
plot(t2, d1(1:length(t2)),'Linewidth',2,'Color','k')
ax=gca;
ax.FontSize = 15;
hold on
plot(t2, d2(1:length(t2)),'Linewidth',2,'Color','r')
legend("$\delta_1$", "$\delta_2$", 'Interpreter','latex','FontSize',25,'Position',[.5 .67 .2 .1],'FontSize',18)
%ylim([0 5])
xlabel("Time (s)")
print('deltac','-depsc')
%%
figure
lenJ=length(t)-1;
plot(t(1:lenJ),Jnd(:,2))
hold on
plot(t(1:lenJ),J(:,2))
plot(t(1:lenJ),ones(lenJ,1)*J2ref,'--','color','black','LineWidth',2);
xlabel('time (s)')
ylabel('$J_2$ ','Interpreter','latex')
legend('$J_2$ with no deception','$J_2$ with deception','$J_2^{ref}$','Interpreter','latex')
ax=gca;
ax.FontSize = 15;
title('Profit for player 2')
print('duo_profit2','-depsc')
%% dynamics of delta
figure 
plot(t,u(:,3),'LineWidth',2,'Color','m')
ax=gca;
ax.FontSize = 15;
xlabel('time (s)')
ylabel('\delta ')
title('Dynamics of \delta')
print('duo_delta','-depsc')
%% players 1 and 2 deceive each other (with compensator)
clf
h=0.0001;                                             
t = 0:h:50;

S=100;
p=0.2;
m1=30;
m2=30;

Q1=2*[-1/p,1/(2*p);1/(2*p),0];
b1=[(m1+S*p)/p,-1*m1/p]';
Q2=2*[0,1/(2*p);1/(2*p),-1/p];
b2=[-1*m2/p,m2/p]';

% Q1=[3 1;1 5];
% b1=[4 2]';
% Q2=[7 2;2 4];
% b2=[1 6]';



A0=[Q1(1,:);Q2(2,:)];
b0=[b1(1),b2(2)]';
x0=-1*inv(A0)*b0;

%c1 and c2
c1=-1*(Q1(1,1)+Q2(2,2))/Q1(1,2);
c2=det(A0)/(Q2(1,2)*Q1(2,2)-Q2(2,2)*Q1(1,2));

%functions
w=[1 -1*Q2(1,2)/Q2(2,2)]';
q1=-1*(b1(2)+Q1(2,:)*x0);
q2=Q1(2,:)*w;
q3=Q1(1,:)*w;
r2=0.5*w'*Q2*w;
r1=(b2+Q2*x0)'*w;



a=0.05;
k=-0.03;
w1=11877.75;
w2=12436.5;
g1_1=3;
g1_2=13;
g2_1=2;
g2_2=10;


e_delta=@(d) d*q1/(d*q2+q3);
J1=@(x) 0.5*x'*Q1*x+b1'*x-3000;
J2=@(x) 0.5*x'*Q2*x+b2'*x;

J2quad=@(e) r2*e^2+r1*e+J2(x0);
%J1=@(x) (x-5).^2;
%J2=@(x) (x-10).^2;

epsilon1=-0.001;
epsilon2=-0.0005;
J2ref=1800;
J1ref=1200;
prices=@(t,u,d1, d2) u+a*[sin(w1*t)+d1*sin(w2*t), sin(w2*t)+(d2)*sin(w1*t)]';
delta1=@(z,e) (g1_2/g1_1).*e-(g1_2/g1_1-1).*z;
delta2=@(z,e) (g2_2/g2_1).*e-(g2_2/g2_1-1).*z;
% u3,4 are z1,e1 and u5,6 are z2,e2
udot=@(time,u) [(-2*k/a)*J1(prices(time,u(1:2),delta1(u(3),u(4)), delta2(u(5),u(6))))*sin(w1*time),(-2*k/a)*J2(prices(time,u(1:2),delta1(u(3),u(4)), delta2(u(5),u(6))))*sin(w2*time),(1/g1_1)*(-u(3)+u(4)), epsilon1*(J1(prices(time,u(1:2),delta1(u(3),u(4)), delta2(u(5),u(6))))-J1ref),(1/g2_1)*(-u(5)+u(6)),epsilon2*(J2(prices(time,u(1:2),delta1(u(3),u(4)), delta2(u(5),u(6))))-J2ref)]';
%udot=@(time,u) [(-2*k/a)*J1(u(1:2)+a*[sin(w1*time),sin(w2*time)+u(3)*sin(w1*time)]')*sin(w1*time),(-2*k/a)*J2(u(1:2)+a*[sin(w1*time),sin(w2*time)+u(3)*sin(w1*time)]')*sin(w2*time),epsilon*(J2(u(1:2)+a*[sin(w1*time),sin(w2*time)+u(3)*sin(w1*time)]')-J2ref)]';
u = zeros(length(t),6);
u(1,:)=[50,100/3,0,0,0,0];
%u(1,:)=[0,0];
J=zeros(length(t)-1,2);

for i=1:(length(t)-1) 
    %u(i,4)=0; 
    k_1 = udot(t(i),u(i,:)');
    k_2 = udot(t(i)+0.5*h,u(i,:)'+0.5*h*k_1);
    k_3 = udot((t(i)+0.5*h),(u(i,:)'+0.5*h*k_2));
    k_4 = udot((t(i)+h),(u(i,:)'+k_3*h));
    u(i+1,:) = u(i,:)' + (1/6)*(k_1+2*k_2+2*k_3+k_4)*h;  % main equation
    %delta=delta+ -1*epsilon*(J2(u(i,:)')-J2ref)*h;
    J(i,1)=J1(u(i,1:2)');
    J(i,2)=J2(u(i,1:2)');
  
end

% xd=-1*inv(A0+u(333334,3)*[Q1(2,:);0 0])*(b0+u(333334,3)*[b1(2);0]);
plot(t,u(:,1),'LineWidth',2)
hold on
plot(t,u(:,2),'LineWidth',2)


% for i=1:2
%     plot(t,ones(1,length(t))*xd(i),'--','color','black','LineWidth',2)
%     hold on
% end
xlabel('time (s)')
ylabel('$x_i$ ','Interpreter','latex')
% legend('$x_1$','$x_2$','DNE','Interpreter','latex')
ax=gca;
ax.FontSize = 15;
title('Price convergence with deception')
print('duo_nodec','-depsc')
%%
clf
jref=[J1ref J2ref];
lenJ=length(t)-1;
plot(t(1:lenJ),J(:,1),'LineWidth',2)
hold on
plot(t(1:lenJ),J(:,2),'LineWidth',2)
hold on
% for i=1:2
%     plot(t,ones(1,length(t))*jref(i),'--','color','black','LineWidth',2)
%     hold on
% end
plot(t,ones(1,length(t))*jref(1),'-','color','black','LineWidth',2)
hold on
plot(t,ones(1,length(t))*jref(2),'--','color','black','LineWidth',2)
hold on
legend('$J_1$', '$J_2$', '$J_1^{ref}$', '$J_2^{ref}$','Position',[.48 .32 .2 .1],'Interpreter','latex','FontSize',22)
ylabel('Profit')
xlabel('Time (s)')
ylim([0 4500])
ax=gca;
ax.FontSize = 15;
%title('Mutual Deception with Compensator')
print('mutualdec_c','-depsc')
%% plots
drng=linspace(-10,25,1000);
dcosts=zeros(1,length(drng));
at=[Q1(2,:);0 0];
bt=[b1(2);0];
dhurwitz=(drng<5/3).*(drng>-7);
dhurwitz=logical(dhurwitz);
for i=1:length(drng)
    dne=-1*inv(A0+drng(i)*at)*(b0+drng(i)*bt);
    dcosts(i)=J2(dne);
end
figure
plot(drng,dcosts);
hold on
% plot(drng(dhurwitz), dcosts(dhurwitz),'r');
% hold on

plot(drng,dcosts+4.8333,'LineWidth',2);
hold on
plot(drng,dcosts-31.64,'LineWidth',2);
hold on
% plot(drng(dhurwitz), dcosts(dhurwitz)+4.8333,'r');
% hold on
% 
% plot(drng(dhurwitz), dcosts(dhurwitz)-31.64,'r');
% hold on
% plot(c1*ones(1000,1),linspace(-50,200,1000),':')
% hold on
% plot(c2*ones(1000,1),linspace(-50,200,1000),':')
% hold on
xregion(c1,c2)

plot(drng,zeros(1,length(drng)),'--','color','black','LineWidth',2)
ylabel('$J_2$ at the DNE','Interpreter','latex')
xlabel('\delta')
legend('$J_2$', '$J_2+4.83$','$J_2 -31.64$','$\Delta$','Interpreter','latex')
ax=gca;
ax.FontSize = 15;
ylim([-50 100]);
xlim([-10 13]);
title('$J_2-J_2^{ref}$','Interpreter','latex')
print('j2shift','-depsc')
%% range of J2 values

singularityDist=0.1;
alpha=J2quad(min(e_delta(c1),-0.5*r1/r2));
beta=J2quad(min(q1/q2,-0.5*r1/r2));
if Q1(1,2)>=0

    if Q1(1,2)*Q2(2,2)>Q1(2,2)*Q2(1,2)
        if c1>c2
            J2range=[beta,alpha];
        else
            J2range=[beta,9999999];
        end
    elseif Q1(1,2)*Q2(2,2)==Q1(2,2)*Q2(1,2)
        
        again=1;
    else
        J2range=[J2quad(-0.5*r1/r2),alpha];
    end
else
    if Q1(1,2)*Q2(2,2)>Q1(2,2)*Q2(1,2)
        J2range=[alpha,9999999];
    elseif Q1(1,2)*Q2(2,2)==Q1(2,2)*Q2(1,2)
        again=1;
    else
        if c1<c2
            J2range=[alpha,beta];
        else
            J2range=[J2quad(-0.5*r1/r2),beta];
        end
    end
end



%% tests
J2min=J2(x0)-(r1^2)/(4*r2);
J2quad(-0.5*r1/r2)
deltaI=0.2;
J2(-1*inv(A0+deltaI*[Q1(2,:);0,0])*[b0(1)+deltaI*b1(2),b0(2)]')
%%


% S=100;
% p=0.2;
% m1=30;
% m2=30;

% Q1=2*[-1/p,1/(2*p);1/(2*p),0];
% b1=[(m1+S*p)/p,-1*m1/p]';
% Q2=2*[0,1/(2*p);1/(2*p),-1/p];
% b2=[-1*m2/p,m2/p]';

Q1=[3 1;1 5];
b1=[4 2]';
Q2=[7 2;2 4];
b2=[1 6]';
del=2;


A0=[Q1(1,:);Q2(2,:)];
b0=[b1(1),b2(2)]';
x0=-1*inv(A0)*b0;
Atild=[Q1(2,:);0 0];
btild=[b1(2) 0]';
sm1=-1*inv(A0)*Atild*x0;
sm2=-1*inv(A0)*btild;
sm3=inv(A0)*Atild*inv(A0)*btild;
adel=trace(inv(A0)*Atild);

%c1 and c2
c1=-1*(Q1(1,1)+Q2(2,2))/Q1(1,2);
c2=det(A0)/(Q2(1,2)*Q1(2,2)-Q2(2,2)*Q1(1,2));

%functions
w=[1 -1*Q2(1,2)/Q2(2,2)]';
q1=-1*(b1(2)+Q1(2,:)*x0);
q2=Q1(2,:)*w;
q3=Q1(1,:)*w;
r2=0.5*w'*Q2*w;
r1=(b2+Q2*x0)'*w;
%%
J1tild=@(x,delta) J1(x)+delta*(0.5*Q1(1,2)*x(1)^2+Q1(2,2)*x(1)*x(2)+b1(2)*x(1));
[X,Y] = meshgrid(30:.25:70);
[x1,x2]=size(X);
J1s=zeros(x1,x2);
J11=zeros(x1,x2);
J12=zeros(x1,x2);
J13=zeros(x1,x2);
J14=zeros(x1,x2);
J15=zeros(x1,x2);
J16=zeros(x1,x2);
sgn=1;
for i=1:x1
    for j=1:x2
        J1s(i,j)=J1([X(i,j) Y(i,j)]');
        J11(i,j)=J1tild([X(i,j) Y(i,j)]',sgn*0.1);
        J12(i,j)=J1tild([X(i,j) Y(i,j)]',sgn*0.2);
        J13(i,j)=J1tild([X(i,j) Y(i,j)]',sgn*0.3);
        J14(i,j)=J1tild([X(i,j) Y(i,j)]',sgn*0.4);
        J15(i,j)=J1tild([X(i,j) Y(i,j)]',sgn*0.5);
        J16(i,j)=J1tild([X(i,j) Y(i,j)]',sgn*0.6);
    end
end
figure
surf(X,Y,J1s,'edgecolor','r', 'FaceColor', [255,0,0]/255, 'FaceAlpha', .5)
xlabel('price for player 1')
ylabel('price for player 2')
zlabel('cost')
hold on
surf(X,Y,J11,'edgecolor','black', 'FaceColor', [255,165,0]/255, 'FaceAlpha', .9)
hold on
surf(X,Y,J12,'edgecolor','black', 'FaceColor', [255,255,0]/255, 'FaceAlpha', .9)
hold on
surf(X,Y,J13,'edgecolor','black', 'FaceColor', [0,128,0]/255, 'FaceAlpha', .9)
hold on
surf(X,Y,J14,'edgecolor','black', 'FaceColor', [0,0,255]/255, 'FaceAlpha', .9)
hold on
surf(X,Y,J15,'edgecolor','black', 'FaceColor', [75,0,130]/255, 'FaceAlpha', .9)
hold on
surf(X,Y,J16,'edgecolor','black', 'FaceColor', [238, 130, 238]/255, 'FaceAlpha', .9)
% colororder([1 100/255 0;1/255 1 200/255])
% legend(["Player 1" "Player 2"])
%% plot the costs
clf
J1tild=@(x,delta) J1(x)+delta*(0.5*Q1(1,2)*x(1)^2+Q1(2,2)*x(1)*x(2)+b1(2)*x(1));
[X,Y] = meshgrid(30:.5:70);
[x1,x2]=size(X);
% X=30:0.5:70;
% Y=30:.5:70;
% x1=length(X);
% x2=length(Y);
J1s=zeros(x1,x2);
J2s=zeros(x1,x2);
delta=0;
for i=1:x1
    for j=1:x2
        J1s(i,j)=J1tild([Y(j) X(i)]',delta);
        J2s(i,j)=J2([Y(j) X(i)]');
        J1s(i,j)=J1tild([X(i,j) Y(i,j)]',delta);
        J2s(i,j)=J2([X(i,j) Y(i,j)]');
    end
end
surf(X,Y,J1s,'FaceColor',"#0072BD", 'FaceAlpha', .7)
hold on
surf(X,Y,J2s,'FaceColor',"#D95319", 'FaceAlpha', .7)
ax=gca;
ax.FontSize = 15;
xlabel('$x_1$','Interpreter','latex','FontSize',30)
ylabel('$x_2$','Interpreter','latex','FontSize',30)
title('$t_1=0$', 'Interpreter', 'latex','FontSize',25)
labelz=zlabel('$J_i$','Interpreter','latex','FontSize',30,'Rotation',0);
labelz.Position(3) = 6500;
labelz.Position(2) = 60;
legend('$\tilde{J}_1$', '$J_2$','Interpreter','latex','FontSize',25,'Position',[0.7324    0.8008    0.1365    0.1631])
zlim([-1000 3000])
view(-60, 50)
legend boxoff  
print('j3d0','-depsc')
%%
figure
for i=1:x1
    for j=1:x2
        J1s(i,j)=J1tild([X(i,j) Y(i,j)]',0.4);
    end
end
surf(X,Y,J1s,'FaceColor',"#0072BD", 'FaceAlpha', .7)
hold on
surf(X,Y,J2s,'FaceColor',"#D95319", 'FaceAlpha', .7)
ax=gca;
ax.FontSize = 15;
xlabel('$x_1$','Interpreter','latex','FontSize',30)
ylabel('$x_2$','Interpreter','latex','FontSize',30)
title('$t_2=12$', 'Interpreter', 'latex','FontSize',25)
labelz=zlabel('$J_i$','Interpreter','latex','FontSize',30,'Rotation',0);
labelz.Position(3) = 6500;
labelz.Position(2) = 60;
legend('$\tilde{J}_1$', '$J_2$','Interpreter','latex','FontSize',25,'Position',[0.7324    0.8008    0.1365    0.1631])
zlim([-1000 3000])
view(-60, 50)
legend boxoff  
print('j3d1','-depsc')
%%
figure
for i=1:x1
    for j=1:x2
        J1s(i,j)=J1tild([X(i,j) Y(i,j)]',0.8);
    end
end
surf(X,Y,J1s,'FaceColor',"#0072BD", 'FaceAlpha', .7)
hold on
surf(X,Y,J2s,'FaceColor',"#D95319", 'FaceAlpha', .7)
ax=gca;
ax.FontSize = 15;
xlabel('$x_1$','Interpreter','latex','FontSize',30)
ylabel('$x_2$','Interpreter','latex','FontSize',30)
title('$t_3=100$', 'Interpreter', 'latex','FontSize',25)
labelz=zlabel('$J_i$','Interpreter','latex','FontSize',30,'Rotation',0);
labelz.Position(3) = 6500;
labelz.Position(2) = 60;
legend('$\tilde{J}_1$', '$J_2$','Interpreter','latex','FontSize',25,'Position',[0.7324    0.8008    0.1365    0.1631])
zlim([-1000 3000])
view(-60, 50)
legend boxoff  
print('j3d2','-depsc')
%% Reaction curves
delta=1.0;
m1=-1*(Q1(1,1)+delta*Q1(1,2))/(Q1(1,2)+delta*Q1(2,2));
z1=-1*(b1(1)+delta*b1(2))/(Q1(1,2)+delta*Q1(2,2));
m2=-1*Q2(1,2)/Q2(2,2);
z2=-1*b2(2)/Q2(2,2);

rc1=@(x,m,z) m*x+z;
rc2=@(x) m2*x+z2;
num=10;
deltaVals=linspace(-1,0.7,num);
cm = hsv(num);

ne=zeros(num,2);
mz=zeros(num,2);
for i=1:num
    delta=deltaVals(i);
    mz(i,1)=-1*(Q1(1,1)+delta*Q1(1,2))/(Q1(1,2)+delta*Q1(2,2));
    mz(i,2)=-1*(b1(1)+delta*b1(2))/(Q1(1,2)+delta*Q1(2,2));
    ne(i,1)=(z2-mz(i,2))/(mz(i,1)-m2);
    ne(i,2)=mz(i,1)*ne(i,1)+mz(i,2);
end

figure
intv=[min(ne(:,1))-10, max(ne(:,1))+40]-10;
% line(intv,[rc1(intv(1),mz(1,1),mz(1,2)),rc1(intv(2),mz(1,1),mz(1,2))], 'Color', 'r');
% hold on
% line(intv,[rc1(intv(1),mz(2,1),mz(2,2)),rc1(intv(2),mz(2,1),mz(2,2))], 'Color', [0.8500 0.3250 0.0980]);
% hold on
% line(intv,[rc1(intv(1),mz(3,1),mz(3,2)),rc1(intv(2),mz(3,1),mz(3,2))], 'Color', 'y');
% hold on
% line(intv,[rc1(intv(1),mz(4,1),mz(4,2)),rc1(intv(2),mz(4,1),mz(4,2))], 'Color', 'g');
% hold on
% line(intv,[rc1(intv(1),mz(5,1),mz(5,2)),rc1(intv(2),mz(5,1),mz(5,2))], 'Color', 'b');
% hold on
% line(intv,[rc1(intv(1),mz(6,1),mz(6,2)),rc1(intv(2),mz(6,1),mz(6,2))], 'Color', 'm');
% hold on

for i=1:num
    line(intv,[rc1(intv(1),mz(i,1),mz(i,2)),rc1(intv(2),mz(i,1),mz(i,2))], 'Color', cm(i,:),'LineWidth',2.5);
    hold on
end
hold on
rcurve2=line(intv,[rc2(intv(1)),rc2(intv(2))], 'Color', 'k','LineWidth',1.3);
hold on
colormap(hsv(num))
c=colorbar('Ticks',[0 0.5 1],'TickLabels',{'-1','-0.15','0.7'});
hold on
contour(X,Y,J2c,[750 750],'Color',[.5 .5 .5],'LineWidth',3)
hold on
contour(X,Y,J2c,[500 500],'Color',[.5 .5 .5],'LineWidth',3)
hold on
contour(X,Y,J2c,[250 250],'Color',[.5 .5 .5],'LineWidth',3)
hold on
% for i=1:num
%     hold on
%     dnes(i)=plot(ne(i,1), ne(i,2), 'Marker', '.', 'MarkerSize', 15, 'Color', 'k');
% end
% legend(dnes,'DNE')

dnes=scatter(ne(:,1), ne(:,2),100,'k','Marker','*');
%legend('DNE')
rotatept=plot(30,10, 'pentagram', 'MarkerSize', 15, 'Color', 'k','MarkerFaceColor','r');
%legend(dnes,'DNE')
ax=gca;
ax.FontSize = 15;
legend([rcurve2 dnes rotatept],{'RC for player 2','DNE','Rotation Point'},'Position',[.5 .22 .2 .25],'FontSize', 17)
ylabel('$x_2$','Rotation',0,'Interpreter','latex','FontSize',20)
xlabel('$x_1$','Rotation',0,'Interpreter','latex','FontSize',20)


%c.Label.String = 'RC for player 1';
ylim([5 50])
xlim([25 75])

print('rcrotate_duo','-depsc')
%%
[X,Y] = meshgrid(25:.01:80, -10:0.01:60);
[x1,x2]=size(X);
% X=30:0.5:70;
% Y=30:.5:70;
% x1=length(X);
% x2=length(Y);
J2c=zeros(x1,x2);
delta=0;
for i=1:x1
    for j=1:x2
        J2c(i,j)=J2([X(i,j) Y(i,j)]');
    end
end
%%
clf
f=@(x,y) x.^2 + y.^2;
[X,Y]=meshgrid(-2:0.001:2);
z=zeros(length(X), length(Y));
for i=1:length(X)
    for j=1:length(Y)
        z(i,j)=1;
    end
end
%%
clf
contour(X,Y,J2c,[500 500])

%%
function dxdt=odefcn(t,x)
    dxdt=zeros(2,1);
    
    Q1=[3 1;1 5];
    b1=[4 2]';
    Q2=[7 2;2 4];
    b2=[1 6]';

    A0=[Q1(1,:);Q2(2,:)];
    b0=[b1(1),b2(2)]';
    delta=0.53;
    A=A0+delta*[Q1(2,:);0 0];
    b=b0+delta*[b1(2);0];
    k=0.05;
    mat=-k*A*x-k*b;
    dxdt(1)=mat(1);
    dxdt(2)=mat(2);
end