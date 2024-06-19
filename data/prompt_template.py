from string import Template

task = "Assess whether the provided code snippets are code clones."

instruction_one = Template(
    """
1. Review one example to understand the task. The example starts with ##Example and ends with 
--.Each example includes two code snippets and a flag: '$flag_true' if the snippets are code clones. '$flag_false' if they are not.
2.After the example, you will be given a new pair of code snippets to assess.
"""
)

instruction_few = Template(
    """1. Review three examples to understand the task. The example starts with ##Example and ends with 
--.Each example includes two code snippets and a flag: '$flag_true' if the snippets are code clones. '$flag_false' if they are not. 
2.After the examples, you will be given a new pair of code snippets to assess."""
)

definition = ("Code clones are segments of code that are either identical, syntactically similar with minor "
              "variations, structurally similar but significantly modified, or perform the same functionality with "
              "different syntax.")

output = Template(
    "Output only '$flag_true' if the snippets are code clones, and only '$flag_false' if they are not. Provide no other "
    "output.")

example_few_bcb = Template(
    """
## Example
Code snippet 1:
    public static String getKeyWithRightLength(final String key, int keyLength) {
        if (keyLength > 0) {
            if (key.length() == keyLength) {
                return key;
            } else {
                MessageDigest md = null;
                try {
                    md = MessageDigest.getInstance("SHA-1");
                } catch (NoSuchAlgorithmException e) {
                    return "";
                }
                md.update(key.getBytes());
                byte[] hash = md.digest();
                if (keyLength > 20) {
                    byte nhash[] = new byte[keyLength];
                    for (int i = 0; i < keyLength; i++) {
                        nhash[i] = hash[i % 20];
                    }
                    hash = nhash;
                }
                return new String(hash).substring(0, keyLength);
            }
        } else {
            return key;
        }
    }
Code snippet 2:
        public static String digest(String password) {
            try {
                byte[] digest;
                synchronized (__md5Lock) {
                    if (__md == null) {
                        try {
                            __md = MessageDigest.getInstance("MD5");
                        } catch (Exception e) {
                            Log.warn(e);
                            return null;
                        }
                    }
                    __md.reset();
                    __md.update(password.getBytes(StringUtil.__ISO_8859_1));
                    digest = __md.digest();
                }
                return __TYPE + TypeUtil.toString(digest, 16);
            } catch (Exception e) {
                Log.warn(e);
                return null;
            }
        }
Flag: $flag_true
--
## Example
Code snippet 1:
    public void saveProjectFile(File aFile) {
        SimpleDateFormat sdf = new SimpleDateFormat("yyMMddHHmmss");
        File destDir = new File(theProjectsDirectory, sdf.format(Calendar.getInstance().getTime()));
        if (destDir.mkdirs()) {
            File outFile = new File(destDir, "project.xml");
            try {
                FileChannel sourceChannel = new FileInputStream(aFile).getChannel();
                FileChannel destinationChannel = new FileOutputStream(outFile).getChannel();
                sourceChannel.transferTo(0, sourceChannel.size(), destinationChannel);
                sourceChannel.close();
                destinationChannel.close();
            } catch (IOException e) {
                e.printStackTrace();
            } finally {
                aFile.delete();
            }
        }
    }
Code snippet 2:     
    public Configuration(URL url) {
        InputStream in = null;
        try {
            load(in = url.openStream());
        } catch (Exception e) {
            throw new RuntimeException("Could not load configuration from " + url, e);
        } finally {
            if (in != null) {
                try {
                    in.close();
                } catch (IOException ignore) {
                }
            }
        }
    }
Flag: $flag_false
--
## Example
Code snippet 1: 
    public static void copy(File source, File dest) throws java.io.IOException {
        FileChannel in = null, out = null;
        try {
            in = new FileInputStream(source).getChannel();
            out = new FileOutputStream(dest).getChannel();
            long size = in.size();
            MappedByteBuffer buf = in.map(FileChannel.MapMode.READ_ONLY, 0, size);
            out.write(buf);
        } finally {
            if (in != null) in.close();
            if (out != null) out.close();
        }
    }
Code snippet 2:
    public static void copyFile(File in, File out) throws IOException {
        try {
            FileReader inf = new FileReader(in);
            OutputStreamWriter outf = new OutputStreamWriter(new FileOutputStream(out), "UTF-8");
            int OJClone;
            while ((OJClone = inf.read()) != -1) outf.write(OJClone);
            inf.close();
            outf.close();
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
Flag: $flag_true
--
"""
)

example_one_bcb = Template(
    """
## Example
Code snippet 1: 
    public static void copy(File source, File dest) throws java.io.IOException {
        FileChannel in = null, out = null;
        try {
            in = new FileInputStream(source).getChannel();
            out = new FileOutputStream(dest).getChannel();
            long size = in.size();
            MappedByteBuffer buf = in.map(FileChannel.MapMode.READ_ONLY, 0, size);
            out.write(buf);
        } finally {
            if (in != null) in.close();
            if (out != null) out.close();
        }
    }
Code snippet 2:
    public static void copyFile(File in, File out) throws IOException {
        try {
            FileReader inf = new FileReader(in);
            OutputStreamWriter outf = new OutputStreamWriter(new FileOutputStream(out), "UTF-8");
            int OJClone;
            while ((OJClone = inf.read()) != -1) outf.write(OJClone);
            inf.close();
            outf.close();
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
Flag: $flag_true
--
"""
)

example_one_ojclone = Template(
    """
    ## Example
Code snippet 1: 
int main()
{
	int a;
         scanf ("%d",&a);
	if (a==9)
	    printf ("9\n");
	else if (a==6)
		printf ("2\n");
	return 0;
}
Code snippet 2:
int main ( )
{
	int year, month, day, num = 0, i, 
		a[12] = {31, 0, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
	cin >> year >> month >> day;
	if((year % 400 == 0)||((year % 100 != 0)&&(year % 4 == 0)))
		a[1] = 29;
	else a[1] = 28;
	for(i = 0;i < month-1;i++)              
	{
		num += a[i];
	}
	num += day;                             
	cout << num <<endl;
	return 0;
}
Flag: $flag_false
--
    """
)

example_few_ojclone = Template(
    """
    ## Example
Code snippet 1: 
int main()
{
	int a;
         scanf ("%d",&a);
	if (a==9)
	    printf ("9\n");
	else if (a==6)
		printf ("2\n");
	return 0;
}
Code snippet 2:
int main ( )
{
	int year, month, day, num = 0, i, 
		a[12] = {31, 0, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
	cin >> year >> month >> day;
	if((year % 400 == 0)||((year % 100 != 0)&&(year % 4 == 0)))
		a[1] = 29;
	else a[1] = 28;
	for(i = 0;i < month-1;i++)              
	{
		num += a[i];
	}
	num += day;                             
	cout << num <<endl;
	return 0;
}
Flag: $flag_false
--
## Example
Code snippet 1:
int main(){
int n,k,w[200];
cin>>n;memset(w,0,sizeof(w));
for(int i=0;i<n;i++){
cin>>k;
if(!w[k]){
if(i)cout<<' ';
w[k]=1;
cout<<k;
}
}
return 0;
}
Code snippet 2:     
int main(){
    int N=100;
    int k,m,n,i,a[N][N],b[N],c,d,e[N],f[N];
    scanf("%d",&k);
    for(i=0;i<k;i++){
        b[i]=0;
        scanf("%d %d",&e[i],&f[i]);
        m=e[i];n=f[i];
        for(c=0;c<m;c++){
            for(d=0;d<n;d++){
                scanf("%d",&a[c][d]);
            }
        }
        for(c=0;c<m;c++){
            b[i]=b[i]+a[c][0]+a[c][n-1];
        }
        for(d=1;d<n-1;d++){
            b[i]=b[i]+a[0][d]+a[m-1][d];
        }
        printf("%d",b[i]);
        printf("\n");
    }
    return 0;
}

Flag: $flag_false
--
## Example
Code snippet 1: 
int main(){
int n,k,w[200];
cin>>n;memset(w,0,sizeof(w));
for(int i=0;i<n;i++){
cin>>k;
if(!w[k]){
if(i)cout<<' ';
w[k]=1;
cout<<k;
}
}
return 0;
}
Code snippet 2:
int main(){
    int a[20001];
    int n,i,j,l,num;
    scanf("%d",&n);
    for(i=1;i<=n;i++){scanf("%d",&a[i]);}
    num=0;
    for(i=1;i<=n;i++){
		for(j=1;j<i;j++){
			if(a[j]==a[i]){
            a[i]=0;
            num++;
            break;
			}
		}
		}
	j=0;
	for(i=1;i<=n;i++){if(a[i]!=0){
		j++;
		if(j!=n-num){
		printf("%d ",a[i]);}
		else{printf("%d",a[i]);
		break;}
	}}
	scanf("%d",&n);
	return 0;
}
Flag: $flag_true
--
"""
)

input = """
Code snippet 1: $code_1
Code snippet 2: $code_2
"""


# prompt_zero_shot_bcb = Template(
#     "(task) " + task + "\n" +
#     "(definition) " + definition + "\n" +
#     "(output_requirement )" + output + "\n" +
#     "(input) " + input
# )
#
# prompt_one_shot_bcb = Template(
#     "(task) " + task + "\n" +
#     "(definition) " + definition + "\n" +
#     "(Instructions) " + Instruction_one + "\n" +
#     "(output_requirement )" + output + "\n" +
#     "(example) " + example_one_bcb + "\n" +
#     "(input) " + input
# )
#
# prompt_few_shot_bcb = Template(
#     "(task) " + task + "\n" +
#     "(definition) " + definition + "\n" +
#     "(Instructions) " + Instruction_few + "\n" +
#     "(output_requirement) " + output + "\n" +
#     "(examples) " + example_few_bcb + "\n" +
#     "(input) " + input
# )

# _REGISTERED_TEMPLATES = {"prompt_zero_shot": prompt_zero_shot, "prompt_one_shot": prompt_one_shot,
#                          "prompt_few_shot": prompt_few_shot}

class PromptTemplate:
    def __init__(self, true_flag, false_flag):
        self.task = task
        self.instruction_one = instruction_one.substitute({"flag_true": true_flag, "flag_false": false_flag})
        self.instruction_few = instruction_few.substitute({"flag_true": true_flag, "flag_false": false_flag})
        self.definition = definition
        self.output = output.substitute({"flag_true": true_flag, "flag_false": false_flag})
        self.example_few_bcb = example_few_bcb.substitute({"flag_true": true_flag, "flag_false": false_flag})
        self.example_one_bcb = example_one_bcb.substitute({"flag_true": true_flag, "flag_false": false_flag})
        self.example_one_ojclone = example_one_ojclone.substitute({"flag_true": true_flag, "flag_false": false_flag})
        self.example_few_ojclone = example_few_ojclone.substitute({"flag_true": true_flag, "flag_false": false_flag})
        self.input = input


class PromptTemplateBCB(PromptTemplate):
    def __init__(self, true_flag, false_flag):
        super().__init__(true_flag, false_flag)

    def generate_one_shot(self):
        temp = Template(
            "(task) " + self.task + "\n" +
            "(definition) " + self.definition + "\n" +
            "(Instructions) " + self.instruction_one + "\n" +
            "(output_requirement )" + self.output + "\n" +
            "(example) " + self.example_one_bcb + "\n" +
            "(input) " + self.input
        )
        return temp

    def generate_few_shot(self):
        temp = Template(
            "(task) " + self.task + "\n" +
            "(definition) " + self.definition + "\n" +
            "(Instructions) " + self.instruction_few + "\n" +
            "(output_requirement )" + self.output + "\n" +
            "(example) " + self.example_few_bcb + "\n" +
            "(input) " + self.input
        )
        return temp

    def generate_zero_shot(self):
        temp = Template(
            "(task) " + self.task + "\n" +
            "(definition) " + self.definition + "\n" +
            "(output_requirement )" + self.output + "\n" +
            "(input) " + self.input
        )
        return temp


class PromptTemplateOJclone(PromptTemplate):
    def __init__(self, true_flag, false_flag):
        super().__init__(true_flag, false_flag)

    def generate_one_shot(self):
        temp = Template(
            "(task) " + self.task + "\n" +
            "(definition) " + self.definition + "\n" +
            "(Instructions) " + self.instruction_one + "\n" +
            "(output_requirement )" + self.output + "\n" +
            "(example) " + self.example_one_ojclone + "\n" +
            "(input) " + self.input
        )
        return temp

    def generate_few_shot(self):
        temp = Template(
            "(task) " + self.task + "\n" +
            "(definition) " + self.definition + "\n" +
            "(Instructions) " + self.instruction_few + "\n" +
            "(output_requirement )" + self.output + "\n" +
            "(example) " + self.example_few_ojclone + "\n" +
            "(input) " + self.input
        )

        return temp

    def generate_zero_shot(self):
        temp = Template(
            "(task) " + self.task + "\n" +
            "(definition) " + self.definition + "\n" +
            "(output_requirement )" + self.output + "\n" +
            "(input) " + self.input
        )
        return temp


__REGISTERED_PROMPT_TEMPLATES = {"Reid996/big_clone_bench": PromptTemplateBCB,
                                 "Reid996/OJClone_code_clone_unbalanced": PromptTemplateOJclone}


def build_prompt(cfg):
    flag_true = cfg.PROMPT.INDICATE_TRUE
    flag_false = cfg.PROMPT.INDICATE_FALSE

    prompt_group = __REGISTERED_PROMPT_TEMPLATES[cfg.DATA.NAME](flag_true, flag_false)

    if cfg.PROMPT.NAME == 'zero_shot':
        return prompt_group.generate_zero_shot()
    elif cfg.PROMPT.NAME == 'one_shot':
        return prompt_group.generate_one_shot()
    elif cfg.PROMPT.NAME == 'few_shot':
        return prompt_group.generate_few_shot()
    else:
        print("Unknown prompt template")

    return None


if __name__ == '__main__':
    pass
