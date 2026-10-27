#include <gtest/gtest.h>
#include <limits>
#include "vamp_node/empty_hand_withdrawal.hpp"

TEST(EmptyWithdrawal, ContactCanShiftInsideFrozenRegionWhileSeparating) {
    vamp_node::WithdrawalContacts c;
    EXPECT_TRUE(c.seed(7,.004F));
    EXPECT_TRUE(c.seed(8,-.001F));  // Original 2 mm finger proximity shell.
    c.begin();
    EXPECT_TRUE(c.permits(7,.003F,.010F,false));
    EXPECT_TRUE(c.permits(8,.0005F,.010F,false));
    EXPECT_TRUE(c.separating(.010F));
    EXPECT_FALSE(c.permits(9,.0001F,.010F,false));
    EXPECT_FALSE(c.permits(7,.007F,.010F,false));
    EXPECT_FALSE(c.permits(7,.001F,.031F,false));
    EXPECT_FALSE(c.permits(7,.001F,.010F,true));
}
TEST(EmptyWithdrawal, RejectsDeepAndInvalidContacts) {
    vamp_node::WithdrawalContacts c;
    EXPECT_FALSE(c.seed(0,.031F));
    EXPECT_FALSE(c.seed(0,-.0021F));
    EXPECT_FALSE(c.seed(0,std::numeric_limits<float>::quiet_NaN()));
    EXPECT_EQ(c.size(),0u);
    EXPECT_TRUE(c.seed(4,.016F));
    c.begin();
    EXPECT_FALSE(c.permits(4,.001F,-.001F,false));
    EXPECT_FALSE(c.permits(4,-.001F,.001F,false));
}
TEST(EmptyWithdrawal, TangentialContactKeepsFixedBudgetAndMustClearAtEndpoint) {
    vamp_node::WithdrawalContacts c;
    EXPECT_TRUE(c.seed(2,.004F));
    for (int step=0;step<25;++step) {
        c.begin();
        EXPECT_TRUE(c.permits(2,.002F,.001F*step,false));
        EXPECT_TRUE(c.separating(.001F*step));
        EXPECT_FALSE(c.permits(3,.001F,.001F*step,false));
    }
    c.begin();
    EXPECT_TRUE(c.permits(2,.004F,.020F,false));
    EXPECT_TRUE(c.separating(.020F));
    // A slowly separating rim was falsely rejected by the old 50%/3 cm rule.
    c.begin();
    EXPECT_TRUE(c.permits(2,.00396F,.002F,false));
    EXPECT_TRUE(c.separating(.002F));
    c.begin();
    EXPECT_TRUE(c.permits(2,.0045F,.002F,false));
    EXPECT_FALSE(c.separating(.002F));
    EXPECT_FALSE(c.permits(2,.001F,.020F,true));
    EXPECT_FALSE(c.permits(2,.001F,.031F,false));
}
TEST(EmptyWithdrawal, InitiallyClearHandGetsNoNewContactAllowance) {
    vamp_node::WithdrawalContacts c;
    EXPECT_TRUE(c.seed(2,-.001F));
    c.begin();
    EXPECT_TRUE(c.permits(2,.0001F,.001F,false));
    EXPECT_FALSE(c.separating(.001F));
    c.begin();
    EXPECT_TRUE(c.separating(.050F));
}
