import { motion } from 'framer-motion'
import { StackVertical } from '@/components/layout/layout-stack/layout-stack'
import Text from '@/components/ui/text/text'
import TextHeading from '@/components/ui/text-heading/text-heading'
import { List, ListItem } from '@/components/ui/list/list'
import Link from 'next/link'

export function WritingSection() {
    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.1, duration: 0.5 }}
        >
            <StackVertical gap="sm">
                <TextHeading as="h2">Writing</TextHeading>
                <Text size="sm">
                    I write about ML, math, and ideas on my{' '}
                    <Link href="/blog" className="text-purple-500 font-medium hover:underline">
                        blog
                    </Link>
                    . Some posts to get you started:
                </Text>
                <List spacing="tight">
                    <ListItem>
                        <Link href="/blog/advancing-language-models" className="text-sm underline hover:text-purple-500">
                            Advancing Language Models: How Transformers and Self-Attention Are Changing the Game
                        </Link>
                    </ListItem>
                    <ListItem>
                        <Link href="/blog/diving-deep-into-h2oai" className="text-sm underline hover:text-purple-500">
                            Diving Deep Into H2O.ai
                        </Link>
                    </ListItem>
                </List>
            </StackVertical>
        </motion.div>
    )
}
