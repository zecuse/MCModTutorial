package zecuse.MCModTutorial.init;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import net.minecraftforge.fml.common.registry.GameRegistry;
import zecuse.MCModTutorial.block.GenericBlock;
import zecuse.MCModTutorial.block.Leaves;
import zecuse.MCModTutorial.reference.Reference;

//This class loads the blocks into the game.
@GameRegistry.ObjectHolder(Reference.MOD_ID)
public class ModBlocks
{
	// This variable holds all of the mod block instances.
	// A List collection is used because the mod doesn't have too many blocks.
	// A Set collection should be used in that event.
	private static final List<GenericBlock> BLOCKS = new ArrayList<>();
	
	// List all of the mod block instances.
	public static final GenericBlock LEAVES = new Leaves();
	
	public static Collection<GenericBlock> getBlocks()
	{
		return BLOCKS;
	}
	
	// Add the mod block to the collection for later registration/usage.
	public static void register(GenericBlock block)
	{
		BLOCKS.add(block);
	}
}
