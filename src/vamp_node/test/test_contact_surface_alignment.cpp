#include "vamp_node/contact_surface_alignment.hpp"
#include <gtest/gtest.h>
#include <limits>
TEST(ContactSurfaceAlignment, CorrectsPerpendicularPadAndKeepsRigidBasis) {
  auto r=vamp_node::alignContactSurfaces(Eigen::Matrix3f::Identity(),
    Eigen::Vector3f::UnitZ(), -Eigen::Vector3f::UnitZ(), .174533F);
  ASSERT_TRUE(r);
  EXPECT_LE(vamp_node::padSurfaceAngle(r->col(1),Eigen::Vector3f::UnitZ(),
    -Eigen::Vector3f::UnitZ()), .174533F);
  EXPECT_NEAR(r->determinant(),1.F,1.e-5F);
}
TEST(ContactSurfaceAlignment, RejectsIncompatibleSurfaces) {
  EXPECT_FALSE(vamp_node::alignContactSurfaces(Eigen::Matrix3f::Identity(),
    Eigen::Vector3f::UnitY(),Eigen::Vector3f::UnitZ(), .349066F));
}
TEST(ContactSurfaceAlignment, PreservesAlreadyAlignedOrientation) {
  auto r=vamp_node::alignContactSurfaces(Eigen::Matrix3f::Identity(),
    Eigen::Vector3f::UnitY(),-Eigen::Vector3f::UnitY(), .174533F);
  ASSERT_TRUE(r); EXPECT_NEAR((*r-Eigen::Matrix3f::Identity()).norm(),0.F,1.e-5F);
}

TEST(ContactSurfaceAlignment, AllowsOrderedWiderToleranceForCurvedContactPair) {
  const auto left=Eigen::AngleAxisf(.22F,Eigen::Vector3f::UnitZ())*Eigen::Vector3f::UnitY();
  const auto right=Eigen::AngleAxisf(-.22F,Eigen::Vector3f::UnitZ())*Eigen::Vector3f::UnitY();
  EXPECT_FALSE(vamp_node::alignContactSurfaces(Eigen::Matrix3f::Identity(),left,right,.174533F));
  EXPECT_TRUE(vamp_node::alignContactSurfaces(Eigen::Matrix3f::Identity(),left,right,.261799F));
}
TEST(ContactSurfaceAlignment, RejectsMissingNormal) {
  EXPECT_FALSE(vamp_node::alignContactSurfaces(Eigen::Matrix3f::Identity(),
    Eigen::Vector3f::Zero(),Eigen::Vector3f::UnitY(),.174533F));
}

TEST(ContactSurfaceAlignment, ThirtyDegreeTrialHandlesWiderSurfaceDifference) {
  const auto left=Eigen::AngleAxisf(.436332F,Eigen::Vector3f::UnitZ())*Eigen::Vector3f::UnitY();
  const auto right=Eigen::AngleAxisf(-.436332F,Eigen::Vector3f::UnitZ())*Eigen::Vector3f::UnitY();
  EXPECT_FALSE(vamp_node::alignContactSurfaces(Eigen::Matrix3f::Identity(),left,right,.174533F));
  EXPECT_FALSE(vamp_node::alignContactSurfaces(Eigen::Matrix3f::Identity(),left,right,.349066F));
  const auto result=vamp_node::alignContactSurfaces(Eigen::Matrix3f::Identity(),left,right,.523599F);
  ASSERT_TRUE(result);
  EXPECT_LE(vamp_node::padSurfaceAngle(result->col(1),left,right),.523599F);
}

TEST(ContactSurfaceAlignment, RejectsCrossingToOppositeApproachSide) {
  const Eigen::Matrix3f original=Eigen::AngleAxisf(.523599F,Eigen::Vector3f::UnitX()).toRotationMatrix();
  const Eigen::Matrix3f opposite=Eigen::AngleAxisf(-.174533F,Eigen::Vector3f::UnitX()).toRotationMatrix();
  EXPECT_FALSE(vamp_node::preservesApproachSide(original,opposite));
  const Eigen::Vector3f normal=opposite.col(1);
  EXPECT_FALSE(vamp_node::alignContactSurfaces(original,normal,-normal,.05F));
}
TEST(ContactSurfaceAlignment, PermitsSameSideAndVerticalCorrections) {
  const Eigen::Matrix3f original=Eigen::AngleAxisf(.523599F,Eigen::Vector3f::UnitX()).toRotationMatrix();
  const Eigen::Matrix3f same_side=Eigen::AngleAxisf(.349066F,Eigen::Vector3f::UnitX()).toRotationMatrix();
  EXPECT_TRUE(vamp_node::preservesApproachSide(original,same_side));
  EXPECT_TRUE(vamp_node::preservesApproachSide(original,Eigen::Matrix3f::Identity()));
  const Eigen::Vector3f normal=same_side.col(1);
  auto aligned=vamp_node::alignContactSurfaces(original,normal,-normal,.05F);
  ASSERT_TRUE(aligned); EXPECT_TRUE(vamp_node::preservesApproachSide(original,*aligned));
}
TEST(ContactSurfaceAlignment, RejectsReversedApproachDirection) {
  const Eigen::Matrix3f reversed=Eigen::AngleAxisf(3.141593F,Eigen::Vector3f::UnitX()).toRotationMatrix();
  EXPECT_FALSE(vamp_node::preservesApproachSide(Eigen::Matrix3f::Identity(),reversed));
}

TEST(ContactSurfaceAlignment, SameApproachSideCanStillMoveHandInward) {
  auto pose=[](float angle) {
    Eigen::Isometry3f p=Eigen::Isometry3f::Identity();
    p.linear()=Eigen::AngleAxisf(angle,Eigen::Vector3f::UnitY()).toRotationMatrix();
    p.translation()=Eigen::Vector3f(.1F,0,0)+p.linear().col(2)*.08F;
    return p;
  };
  const auto original=pose(-.4F),inward=pose(-.2F),outward=pose(-.6F);
  EXPECT_TRUE(vamp_node::preservesApproachSide(original.linear(),inward.linear()));
  EXPECT_FALSE(vamp_node::handAdjustmentIsOutward(original,inward,Eigen::Vector3f(.1F,0,0),Eigen::Vector3f::Zero(),.14F));
  EXPECT_TRUE(vamp_node::handAdjustmentIsOutward(original,outward,Eigen::Vector3f(.1F,0,0),Eigen::Vector3f::Zero(),.14F));
}
TEST(ContactSurfaceAlignment, OutwardUsesObjectPositionNotWorldOrigin) {
  auto before=Eigen::Isometry3f::Identity(),after=before;
  before.translation()=Eigen::Vector3f(.5F,.2F,.2F);
  after=before;after.translation().x()-=.005F;
  EXPECT_TRUE(vamp_node::handAdjustmentIsOutward(before,after,
    Eigen::Vector3f(.5F,.2F,0),Eigen::Vector3f(.6F,.2F,0),.14F));
  EXPECT_FALSE(vamp_node::handAdjustmentIsOutward(before,after,
    Eigen::Vector3f(.5F,.2F,0),Eigen::Vector3f(.4F,.2F,0),.14F));
}
TEST(ContactSurfaceAlignment, AmbiguousObjectSideCannotCertifyCorrection) {
  const auto pose=Eigen::Isometry3f::Identity();
  EXPECT_FALSE(vamp_node::handAdjustmentIsOutward(pose,pose,
    Eigen::Vector3f::Zero(),Eigen::Vector3f::Zero(),.14F));
}

TEST(ContactSurfaceAlignment, RejectedPreviewNeverBecomesValidTarget) {
  Eigen::Matrix3f preview=Eigen::Matrix3f::Identity();
  const auto accepted=vamp_node::alignContactSurfaces(Eigen::Matrix3f::Identity(),
    Eigen::Vector3f::UnitY(),Eigen::Vector3f::UnitZ(),.174533F,&preview);
  EXPECT_FALSE(accepted);
  EXPECT_TRUE(preview.allFinite());
  EXPECT_NEAR(preview.determinant(),1.F,1.e-5F);
  EXPECT_GT(vamp_node::padSurfaceAngle(preview.col(1),
    Eigen::Vector3f::UnitY(),Eigen::Vector3f::UnitZ()),.174533F);
}

TEST(ContactSurfaceAlignment, OutwardBendUsesOriginalHingeAndFixedContactPivot) {
  Eigen::Isometry3f original=Eigen::Isometry3f::Identity();
  const Eigen::Vector3f contact(0,-.1F,0),center=Eigen::Vector3f::Zero();
  original.translation()=contact+Eigen::Vector3f(0,0,.08F);
  float previous_bend=10.F;
  for (float degrees : {10.F,20.F,30.F}) {
    const float radians=degrees*3.14159265F/180.F;
    auto pose=vamp_node::bendOriginalGraspOutward(original,Eigen::Vector3f::UnitZ(),
      contact,center,.14F,.06F,radians);
    ASSERT_TRUE(pose);
    EXPECT_NEAR(vamp_node::fingerSurfaceAngle(pose->linear(),Eigen::Vector3f::UnitZ()),radians,1.e-5F);
    EXPECT_NEAR((pose->linear().col(0)-original.linear().col(0)).norm(),0.F,1.e-6F);
    EXPECT_NEAR((pose->translation()-pose->linear().col(2)*.08F-contact).norm(),0.F,1.e-6F);
    EXPECT_TRUE(vamp_node::handAdjustmentIsOutward(original,*pose,contact,center,.14F));
    const float bend=Eigen::AngleAxisf(original.linear().transpose()*pose->linear()).angle();
    EXPECT_LT(bend,previous_bend);previous_bend=bend;
  }
}
TEST(ContactSurfaceAlignment, OppositeObjectSideReversesHingeBend) {
  auto make=[](float side) {
    Eigen::Isometry3f original=Eigen::Isometry3f::Identity();
    const Eigen::Vector3f contact(0,side*.1F,0);
    original.translation()=contact+Eigen::Vector3f(0,0,.08F);
    return vamp_node::bendOriginalGraspOutward(original,Eigen::Vector3f::UnitZ(),
      contact,Eigen::Vector3f::Zero(),.14F,.06F,.174533F);
  };
  const auto left=make(-1),right=make(1);ASSERT_TRUE(left);ASSERT_TRUE(right);
  EXPECT_NEAR(left->linear()(1,2),-right->linear()(1,2),1.e-6F);
}

TEST(ContactSurfaceAlignment, DynamicSweepOnlyUsesImprovingTargets) {
  const float rad=3.14159265358979323846F/180.F;
  EXPECT_EQ(vamp_node::outwardBendTargets(90.F*rad),(std::vector<float>{10.F,30.F,50.F}));
  EXPECT_EQ(vamp_node::outwardBendTargets(45.F*rad),(std::vector<float>{10.F,30.F}));
  EXPECT_EQ(vamp_node::outwardBendTargets(20.F*rad),(std::vector<float>{10.F}));
  EXPECT_TRUE(vamp_node::outwardBendTargets(5.F*rad).empty());
  EXPECT_TRUE(vamp_node::outwardBendTargets(10.F*rad).empty());
  EXPECT_EQ(vamp_node::outwardBendTargets(30.F*rad),(std::vector<float>{10.F}));
  EXPECT_EQ(vamp_node::outwardBendTargets(50.F*rad),(std::vector<float>{10.F,30.F}));
}
TEST(ContactSurfaceAlignment, DynamicSweepSkipsInvalidAngle) {
  EXPECT_TRUE(vamp_node::outwardBendTargets(std::numeric_limits<float>::quiet_NaN()).empty());
}

TEST(ContactSurfaceAlignment, RecentersOnlyDepthWithoutRelaxingPadBounds) {
  auto pose=Eigen::Isometry3f::Identity();
  const std::array<Eigen::Vector3f,2> contacts{{
    Eigen::Vector3f(0,.009F,.07395F+.0615F),
    Eigen::Vector3f(0,-.009F,.06430F+.0615F)}};
  const auto advance=vamp_node::contactPadAdvance(pose,contacts,.018F,.070F,.020F,-.0615F);
  ASSERT_TRUE(advance);EXPECT_NEAR(*advance,.00695F,1.e-6F);
  pose.translation().z()+=*advance;
  const auto final=vamp_node::contactPadAdvance(pose,contacts,.018F,.070F,.020F,-.0615F);
  ASSERT_TRUE(final);EXPECT_FLOAT_EQ(*final,0.F);
}
TEST(ContactSurfaceAlignment, AdvanceCannotRepairLateralContactError) {
  const std::array<Eigen::Vector3f,2> contacts{{Eigen::Vector3f(.02F,.009F,.07F),Eigen::Vector3f(0,-.009F,.06F)}};
  EXPECT_FALSE(vamp_node::contactPadAdvance(Eigen::Isometry3f::Identity(),contacts,.018F,.07F,.02F,0.F));
}
TEST(ContactSurfaceAlignment, AdvanceCannotExceedBudgetOrLoseOtherContact) {
  const auto pose=Eigen::Isometry3f::Identity();
  std::array<Eigen::Vector3f,2> contacts{{Eigen::Vector3f(0,.009F,.080F),Eigen::Vector3f(0,-.009F,.060F)}};
  EXPECT_FALSE(vamp_node::contactPadAdvance(pose,contacts,.018F,.07F,.02F,0.F));
  contacts[0].z()=.074F;contacts[1].z()=.003F;
  EXPECT_FALSE(vamp_node::contactPadAdvance(pose,contacts,.018F,.07F,.02F,0.F));
}
TEST(ContactSurfaceAlignment, ValidPadsNeedNoAdvance) {
  const std::array<Eigen::Vector3f,2> contacts{{Eigen::Vector3f(0,.009F,.065F),Eigen::Vector3f(0,-.009F,.060F)}};
  const auto advance=vamp_node::contactPadAdvance(Eigen::Isometry3f::Identity(),contacts,.018F,.07F,.02F,0.F);
  ASSERT_TRUE(advance);EXPECT_FLOAT_EQ(*advance,0.F);
}

#include "vamp_node/shallow_grasp.hpp"

TEST(ShallowGrasp, FitsLocalTableAndRejectsUnsupportedPatch) {
  std::vector<Eigen::Vector3f> scene;
  for (int x=-10;x<=10;++x) for (int y=-10;y<=10;++y)
    scene.emplace_back(x*.01F,y*.01F,.02F+x*.0002F);
  for (int i=0;i<50;++i) scene.emplace_back(.02F,.03F,.10F);
  const auto plane=vamp_node::fitGraspSupportPlane(scene,Eigen::Vector3f::Zero());
  ASSERT_TRUE(plane);
  EXPECT_NEAR(plane->normal.dot(Eigen::Vector3f(0,0,.02F)),plane->offset,.0001F);
  EXPECT_NEAR(plane->normal.x(),-.02F,.001F);
  EXPECT_FALSE(vamp_node::fitGraspSupportPlane({Eigen::Vector3f::Zero()},Eigen::Vector3f::Zero()));
}
TEST(ShallowGrasp, OpenFingersMustAlsoClearTable) {
  auto pose=Eigen::Isometry3f::Identity();
  pose.linear()=Eigen::AngleAxisf(.7F,Eigen::Vector3f::UnitX()).toRotationMatrix();
  pose.translation().z()=.01F;
  const vamp_node::GraspSupportPlane plane;
  const float closed=vamp_node::fingerSupportClearance(pose,plane,.02F,.07F,.01F,.02F,0.F);
  const float opened=vamp_node::fingerSupportClearance(pose,plane,.04F,.07F,.01F,.02F,0.F);
  EXPECT_GT(closed,opened);
  EXPECT_LT(opened,0.F);
}
TEST(ShallowGrasp, BoundedShallowerTargetsKeepMeasuredPairInsidePads) {
  auto pose=Eigen::Isometry3f::Identity();pose.translation().z()=.02F;
  const std::array<Eigen::Vector3f,2> contacts{{
    Eigen::Vector3f(0,-.015F,.06F),Eigen::Vector3f(0,.015F,.06F)}};
  const vamp_node::GraspSupportPlane plane;
  const auto targets=vamp_node::shallowContactTargets(pose,contacts,.03F,.07F,.01F,.02F,0.F,plane);
  ASSERT_FALSE(targets.empty());EXPECT_LE(targets.size(),3U);
  for (const auto &target:targets) {
    EXPECT_LE((target.translation()-pose.translation()).norm(),.020001F);
    EXPECT_TRUE(vamp_node::measuredPadContactsSupported(target,contacts,.03F,.07F,.02F,0.F));
    EXPECT_GE(vamp_node::fingerSupportClearance(target,plane,.05F,.07F,.01F,.02F,0.F),.00399F);
  }
}
TEST(ShallowGrasp, RejectsContactsBeyondTipAndLateralMiss) {
  const auto pose=Eigen::Isometry3f::Identity();
  std::array<Eigen::Vector3f,2> contacts{{Eigen::Vector3f(0,-.015F,.073F),Eigen::Vector3f(0,.015F,.069F)}};
  EXPECT_FALSE(vamp_node::measuredPadContactsSupported(pose,contacts,.03F,.07F,.02F,0.F));
  contacts[0]=Eigen::Vector3f(.025F,-.015F,.06F);
  EXPECT_FALSE(vamp_node::measuredPadContactsSupported(pose,contacts,.03F,.07F,.02F,0.F));
}
TEST(ShallowGrasp, RejectsRecordedOriginalFallbackBelowTable) {
  auto pose=Eigen::Isometry3f::Identity();
  pose.translation()=Eigen::Vector3f(.35271907F,.06432456F,.11153939F);
  pose.linear()=Eigen::Quaternionf(.27259481F,.38104868F,.86512643F,.17902569F).normalized().toRotationMatrix();
  const vamp_node::GraspSupportPlane plane{
    Eigen::Vector3f(.02374967F,.0000165753F,.99971794F),-.00246967F};
  const std::array<Eigen::Vector3f,2> contacts{{
    Eigen::Vector3f(.44960317F,.08600911F,.01868500F),
    Eigen::Vector3f(.41806194F,.05234218F,-.00555555F)}};
  EXPECT_NEAR(vamp_node::fingerSupportClearance(pose,plane,.07211444F,.07F,.01F,.02F,-.0615F),-.007171F,.00001F);
  EXPECT_FALSE(vamp_node::measuredPadContactsSupported(pose,contacts,.05211444F,.07F,.02F,-.0615F));
  for (const auto &target : vamp_node::shallowContactTargets(
      pose,contacts,.05211444F,.07F,.01F,.02F,-.0615F,plane)) {
    EXPECT_GE(vamp_node::fingerSupportClearance(target,plane,.07211444F,.07F,.01F,.02F,-.0615F),.00399F);
    EXPECT_TRUE(vamp_node::measuredPadContactsSupported(target,contacts,.05211444F,.07F,.02F,-.0615F));
  }
}
