#include "TouchControl.h"

TouchControl::TouchControl(XF* xf)
{
	this->xf = xf;

}

void TouchControl::touchBegin(Vec2 pos){
	this->beginPos = pos;
}

void TouchControl::touchEnd(Vec2 pos){
	this->endPos = pos;
	const int TURN_LEFT=1;
    const int TURN_RIGHT=2;
	const int UP=3;
	const int DOWN=4;
	float offsetX=endPos.x-beginPos.x;  
    float offsetY=endPos.y-beginPos.y;    
    if(fabs(offsetX)>fabs(offsetY)){//根据X方向与Y方向的偏移量大小的判断  
        if(offsetX<0){  
            if(xf->getXF()->getPositionX()>-10 && !xf->getXF()->getActionByTag(TURN_RIGHT) && !xf->getXF()->getActionByTag(TURN_LEFT)&&!xf->getXF()->getActionByTag(UP))
            {
				auto act = MoveTo::create(0.2,xf->getXF()->getPosition3D()+Vec3(-10,0,0));
				act->setTag(TURN_LEFT);
                this->xf->getXF()->runAction(act);
            }
        }  
        else{  
             if(xf->getXF()->getPositionX()<10 && !xf->getXF()->getActionByTag(TURN_RIGHT) && !xf->getXF()->getActionByTag(TURN_LEFT)&&!xf->getXF()->getActionByTag(UP))
            {
			   auto act = MoveTo::create(0.2,xf->getXF()->getPosition3D()+Vec3(10,0,0));
			   act->setTag(TURN_RIGHT);
               this->xf->getXF()->runAction(act);
            }
        }  
    }  
    else{  
        if(offsetY>0){  
            if(xf->getXF()->getPositionY()<10&& !xf->getXF()->getActionByTag(TURN_RIGHT) && !xf->getXF()->getActionByTag(TURN_LEFT)&&!xf->getXF()->getActionByTag(UP)){
			auto action = MoveTo::create(0.2,Vec3(xf->getXF()->getPositionX(),10,xf->getXF()->getPositionZ()));
			auto action1 = MoveTo::create(0.2,Vec3(xf->getXF()->getPositionX(),0,xf->getXF()->getPositionZ()));
			auto act = Sequence::create(action,action1,NULL);
			act->setTag(UP);
			xf->getXF()->runAction(act);
			}
        }  
    }   
}
