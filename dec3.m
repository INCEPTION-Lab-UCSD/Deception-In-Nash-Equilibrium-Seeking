%clf
% 4 player deception; player 4 is attacking player 1
h=0.0001;                                             
t = 0:h:1500;

mul1=1;
mul2=1;
mul3=1;

Q1=[7 3 1;2 6 -2;-3 3 9];
Q1=.1*mul1*1/2*(Q1+Q1');
Q2=[7 -2 -3;-1 8 1;4 -3 2];
Q2=.1*mul2*1/2*(Q2+Q2');
Q3=[-3 2 2 ;-2 2 4 ;3 -2 7];
Q3=.05*mul3*1/2*(Q3+Q3');

b1=mul1*[2 2 -3]';
b2=mul2*[-1 -3 3]';
b3=mul3*[2 7 -3]';


A0=[Q1(1,:);Q2(2,:);Q3(3,:)];
b0=[b1(1) b2(2) b3(3)]';
x0=-1*inv(A0)*b0;
Qd=@(d) A0+d*[0 0 0;0 0 0;Q3(1,:)];
bd=@(d) b0+d*[0;0;b3(1)];
dne=@(d)-1*inv(Qd(d))*bd(d);

a=0.04;
k=0.02;
c=-2*k/a;

w=4;
w1=w*793.2;
w2=w*511.1;
w3=w*764.4;

J1=@(x) 0.5*x'*Q1*x+b1'*x;
J2=@(x) 0.5*x'*Q2*x+b2'*x;
J3=@(x) 0.5*x'*Q3*x+b3'*x;

Jdref=5;
epsilon=0.001;

prices=@(t,u, d) u+a*[sin(w1*t)+d*sin(w3*t), sin(w2*t), sin(w3*t)]';
udot=@(t,u) c*[J1(prices(t,u(1:3),u(4)))*sin(w1*t), J2(prices(t,u(1:3),u(4)))*sin(w2*t), J3(prices(t,u(1:3),u(4)))*sin(w3*t),(1/c)*epsilon*(J1(prices(t,u(1:3),u(4)))-Jdref)]';

phi1=A0;
phi1(3,:)=[];
phi2=phi1;
phi1(:,1)=[];
phi=[1;-1*inv(phi1)*phi2(:,1)];

q1=-1*(b3(1)+Q3(1,:)*x0);
q2=Q3(1,:)*phi;
q3=Q3(3,:)*phi;

r32=0.5*phi'*Q3*phi;
r31=(Q3*x0+b3)'*phi;
r30=J3(x0);

r22=0.5*phi'*Q2*phi;
r21=(Q2*x0+b2)'*phi;
r20=J2(x0);

r12=0.5*phi'*Q1*phi;
r11=(Q1*x0+b1)'*phi;
r10=J1(x0);

J1q=@(x) r12*x^2+r11*x+r10;
J1min=J1q(-.5*r11/r12);
u = zeros(length(t),4);
%u(1,:)=[x0;0]';
J=zeros(length(t)-1,3);

for i=1:(length(t)-1) 
    %u(i,4)=0; 
    k_1 = udot(t(i),u(i,:)');
    k_2 = udot(t(i)+0.5*h,u(i,:)'+0.5*h*k_1);
    k_3 = udot((t(i)+0.5*h),(u(i,:)'+0.5*h*k_2));
    k_4 = udot((t(i)+h),(u(i,:)'+k_3*h));
    u(i+1,:) = u(i,:)' + (1/6)*(k_1+2*k_2+2*k_3+k_4)*h;  % main equation
    %delta=delta+ -1*epsilon*(J2(u(i,:)')-J2ref)*h;
    
    J(i,1)=J1(u(i,1:3)');
    J(i,2)=J2(u(i,1:3)');
    J(i,3)=J3(u(i,1:3)');
  
end
Jd=J;
for i=1:(length(t)-1) 
    u(i,4)=0; 
    k_1 = udot(t(i),u(i,:)');
    k_2 = udot(t(i)+0.5*h,u(i,:)'+0.5*h*k_1);
    k_3 = udot((t(i)+0.5*h),(u(i,:)'+0.5*h*k_2));
    k_4 = udot((t(i)+h),(u(i,:)'+k_3*h));
    u(i+1,:) = u(i,:)' + (1/6)*(k_1+2*k_2+2*k_3+k_4)*h;  % main equation
    %delta=delta+ -1*epsilon*(J2(u(i,:)')-J2ref)*h;
    
    J(i,1)=J1(u(i,1:3)');
    J(i,2)=J2(u(i,1:3)');
    J(i,3)=J3(u(i,1:3)');
  
end

plot(t,u(:,1),'LineWidth',2)
hold on
plot(t,u(:,2),'LineWidth',2)
hold on
plot(t,u(:,3),'LineWidth',2)
ax=gca;
ax.FontSize = 15;
legend('$x_1$', '$x_2$', '$x_3$','Interpreter','latex','FontSize',20)

%%
clf
lenJ=length(t)-1;
step=50000;
jplot(1)=plot(t(1:step:lenJ),J(1:step:lenJ,1),':','LineWidth',3);
hold on
jplot(2)=plot(t(1:step:lenJ),J(1:step:lenJ,2),':','LineWidth',3);
hold on
jplot(3)=plot(t(1:step:lenJ),J(1:step:lenJ,3),':','LineWidth',3);
hold on
jdplot(1)=plot(t(1:lenJ),Jd(1:lenJ,1),'Linewidth',3,'Color',"#0072BD");
hold on
jdplot(2)=plot(t(1:lenJ),Jd(1:lenJ,2),'Linewidth',3,'Color',"#D95319");
hold on
jdplot(3)=plot(t(1:lenJ),Jd(1:lenJ,3),'Linewidth',3,'Color',"#EDB120");
hold on
jne=plot(t,ones(1,length(t))*J1(x0),'--','color','black','LineWidth',1.2);
hold on
plot(t,ones(1,length(t))*J2(x0),'--','color','black','LineWidth',1.2);
hold on
plot(t,ones(1,length(t))*J3(x0),'--','color','black','LineWidth',1.2);

hold on
d=0.454;
jdne=plot(t,ones(1,length(t))*J1(dne(d)),'-','color','black','LineWidth',2);
hold on
plot(t,ones(1,length(t))*J2(dne(d)),'-','color','black','LineWidth',2);
hold on
plot(t,ones(1,length(t))*J3(dne(d)),'-','color','black','LineWidth',2);
%title('Three Players')
ax=gca;
ax.FontSize = 15;
xlabel('Time (s)')
labelj=ylabel('$J_i$','Interpreter','latex','Rotation',0,'FontSize',20);
ah1 = axes('position',get(gca,'position'),'visible','off');
legend(ah1,jplot(:),'$J_1$', '$J_2$','$J_3$','Position',[.33 .17 .17 .08],'Interpreter','latex','FontSize',13)
ah2 = axes('position',get(gca,'position'),'visible','off');
legend(ah2,jdplot(:),'$J_1$ with deception', '$J_2$ with deception','$J_3$ with deception','Position',[.6 .17 .2 .08],'Interpreter','latex','FontSize',13)
ah3 = axes('position',get(gca,'position'),'visible','off');
legend(ah3, jne, '$J_i(x^*)$','Interpreter','latex','FontSize',15,'Position',[.45 .7 .17 .08])
ah4 = axes('position',get(gca,'position'),'visible','off');
legend(ah4, jdne, '$J_i(x_\delta)$','Interpreter','latex','FontSize',15,'Position',[.65 .7 .17 .08])


%labelj.Position(1) = -45;
print('3ex','-depsc')