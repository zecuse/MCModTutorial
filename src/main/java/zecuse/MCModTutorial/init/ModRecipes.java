package zecuse.MCModTutorial.init;

import net.minecraft.item.ItemStack;
import net.minecraftforge.fml.common.registry.GameRegistry;
import net.minecraftforge.oredict.ShapedOreRecipe;

// This class creates and registers all of the recipes in the mod.
public class ModRecipes
{
	public static void init()
	{
		// The following two recipes do the same thing, but the second allows for any stick and leaf block types.
//		GameRegistry.addShapedRecipe(new ItemStack(ModItems.LEAF, 18), "sls", "lsl", "sls", 's', new ItemStack(Items.STICK), 'l', new ItemStack(Blocks.LEAVES));
		GameRegistry.addRecipe(new ShapedOreRecipe(new ItemStack(ModItems.LEAF, 18), "sls", "lsl", "sls", 's', "stickWood", 'l', "treeLeaves"));
		GameRegistry.addShapedRecipe(new ItemStack(ModBlocks.LEAVES), "lll", "lll", "lll", 'l', new ItemStack(ModItems.LEAF));
		GameRegistry.addShapelessRecipe(new ItemStack(ModItems.LEAF, 9), new ItemStack(ModBlocks.LEAVES));
	}
}
