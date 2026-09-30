"""CPU regression checks. No HCP data/checkpoints or torch required."""
import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from vrptr_clinical import metrics as m
from vrptr_clinical import training_template as t
from vrptr_clinical.compile_individualization import _load_model_tables
from vrptr_clinical.export_revision_audit import audit_selective_gain

class TemplateTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(72)
        self.shape=(1,4,5,6)
        self.template=self.rng.normal(size=self.shape).astype('f4')
        self.truth=[self.template+self.rng.normal(size=self.shape) for _ in range(4)]
        self.pred=[x+self.rng.normal(scale=.4,size=self.shape) for x in self.truth]
        self.ids=['e1','e2','e3','e4']
        self.left=np.arange(60);self.right=np.arange(60,120)
        self.meta={'template_sha256':'abc','template_n_training':3,'target_space':t.TARGET_SPACE}
    def evaluate(self,truth=None,pred=None):
        return m.template_residual_analysis_rows(self.ids,truth or self.truth,pred or self.pred,self.left,self.right,'controls','model',training_template=self.template,template_metadata=self.meta)
    def test_fixed_template_is_independent_of_other_test_targets(self):
        before=self.evaluate()
        modified=[x.copy() for x in self.truth];modified[1]*=100
        after=self.evaluate(truth=modified)
        for key in ['template_pearson_r','delta_pearson_r','residual_pearson_r','template_mae']:
            self.assertEqual(before[0][0][key],after[0][0][key])
    def test_residual_pair_matrix_uses_same_reference(self):
        rows=self.evaluate()[5]
        for row in rows:
            i=self.ids.index(row['subject_pred']);j=self.ids.index(row['subject_true'])
            expected=m.pearsonr_np(self.pred[i]-self.template,self.truth[j]-self.template)
            self.assertAlmostEqual(row['pearson'],expected,places=6)
    def test_point_metrics_unchanged_by_reference(self):
        output=self.evaluate()[0]
        for truth,pred,row in zip(self.truth,self.pred,output):
            self.assertAlmostEqual(row['model_pearson_r'],m.pearsonr_np(truth,pred),places=7)
            self.assertAlmostEqual(row['model_mae'],np.abs(pred-truth).mean(),places=6)
            self.assertAlmostEqual(row['model_rmse'],np.sqrt(((pred-truth)**2).mean()),places=6)
    def test_missing_reference_and_shape_fail(self):
        with self.assertRaises(ValueError):
            m.template_residual_analysis_rows(self.ids,self.truth,self.pred,self.left,self.right,'controls','m')
        self.template=np.ones((1,120))
        with self.assertRaises(ValueError):self.evaluate()
    def test_no_unsupported_eval_voxels(self):
        self.template[0,0,0,0]=np.nan
        with self.assertRaises(ValueError):self.evaluate()
    def test_stream_mean_keeps_missing_support(self):
        a=np.array([1.,np.nan,np.nan]);b=np.array([3.,5.,np.nan])
        mean,count,n=t.mean_training_maps(iter([a,b]))
        np.testing.assert_allclose(mean,[2.,5.,np.nan]);np.testing.assert_array_equal(count,[2,1,0]);self.assertEqual(n,2)
    def test_provenance_grid_overlap_and_checksum(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'template.npz';affine=np.eye(4)
            t.save_template(path,self.template,np.ones(self.shape,dtype='i4'),affine,{'training_subjects':['train1'],'n_training':1})
            loaded,meta=t.read_template(path,self.ids,self.shape,affine)
            np.testing.assert_array_equal(loaded,self.template)
            with self.assertRaises(ValueError):t.read_template(path,['train1'])
            with self.assertRaises(ValueError):t.read_template(path,self.ids,(2,3),affine)
            with self.assertRaises(ValueError):t.read_template(path,self.ids,self.shape,affine*2)
            with np.load(path) as z:payload={k:z[k] for k in z.files}
            payload['template']=payload['template']+1
            np.savez(path,**payload)
            with self.assertRaises(ValueError):t.read_template(path)
    def test_duplicate_and_incomplete_lists(self):
        with self.assertRaises(ValueError):t.validate_subjects(['a','a'],'train')
        with self.assertRaises(ValueError):t.require_complete_subjects(['a','b'],['a'])
    def test_loo_never_fills_with_self(self):
        with self.assertRaises(ValueError):m.leave_one_out_template_maps([self.truth[0]])
        values=m.leave_one_out_template_maps([np.array([1.,2.]),np.array([np.nan,4.])])
        self.assertTrue(np.isnan(values[0][0]));self.assertEqual(values[0][1],4)
    def test_tied_matching_is_not_perfect_identification(self):
        rows,_,_=m.self_vs_other_rows(['a','b'],[self.truth[0]]*2,[self.truth[0]]*2,'controls','m')
        self.assertEqual([r['top1_acc'] for r in rows],[0.,0.])
    def test_compiler_rejects_legacy_and_mixed_templates(self):
        valid=pd.DataFrame(self.evaluate()[0]);self.assertEqual(t.validate_result_frames([valid]),'abc')
        with self.assertRaises(ValueError):t.validate_result_frames([valid.drop(columns=['template_source'])])
        altered=valid.copy();altered['template_sha256']='different'
        with self.assertRaises(ValueError):t.validate_result_frames([valid,altered])
        with self.assertRaises(ValueError):t.validate_result_frames([valid,valid.iloc[:-1]])
    def test_table_merge_retains_provenance(self):
        outputs=self.evaluate()
        whole,_,_=m.self_vs_other_rows(self.ids,self.truth,self.pred,'controls','model')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            for name,rows in [('signal_recovery',outputs[0]),('residual_self_vs_other',outputs[4]),('self_vs_other',whole)]:pd.DataFrame(rows).to_csv(p/(name+'.csv'),index=False)
            frame=_load_model_tables(p)
            self.assertEqual(len(frame),4);self.assertIn('model_mae',frame);self.assertEqual(t.validate_result_frames([frame]),'abc')
    def test_selective_integrated_gain_definition(self):
        risk=pd.DataFrame({'subject':['e1']*3,'condition':['clean']*3,'source':['full_vae_latent']*3,'coverage':[.25,.5,1.],'mae':[.5,1.,2.],'random_expected_mae':[2.]*3})
        reported=pd.DataFrame({'subject':['e1'],'condition':['clean'],'source':['full_vae_latent'],'selective_gain_fraction':[.375]})
        result=audit_selective_gain(risk,reported);self.assertAlmostEqual(result.integrated_gain_fraction.iloc[0],.375)
        reported['selective_gain_fraction']=.2
        with self.assertRaises(ValueError):audit_selective_gain(risk,reported)
    def test_swap_residual_formula(self):
        # Load only the NumPy metric functions from the real GPU module.
        source=Path(__file__).resolve().parents[1]/'vrptr_clinical/skip_bottleneck_swap_analysis.py'
        tree=ast.parse(source.read_text());nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ['_finite_rmse','_pair_row']]
        env={'np':np,'subject_metrics':m.subject_metrics,'pearsonr_np':m.pearsonr_np}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),env)
        kw=dict(variant='full',condition='swap',recipient_subject='e1',donor_subject='e2',true_recipient=self.truth[0],true_donor=self.truth[1],pred=self.pred[1],baseline_recipient=self.pred[0],baseline_donor=self.pred[1],common_template=self.template,left_idx=self.left,right_idx=self.right)
        row=env['_pair_row'](**kw)
        r=lambda a,b:m.pearsonr_np(a-self.template,b-self.template)
        expected=(r(self.pred[1],self.truth[1])-r(self.pred[1],self.truth[0]))-(r(self.pred[0],self.truth[1])-r(self.pred[0],self.truth[0]))
        self.assertAlmostEqual(row['residual_identity_shift_toward_donor'],expected,places=6)
    def test_figure_preserves_mask_and_geometry(self):
        from vrptr_clinical import make_vrptr_current_paper_figures as fig
        a=self.truth[0].copy();a[0,0,0,0]=np.nan
        self.assertTrue(np.isnan(fig._as_volume(a)[0,0,0]))
        self.assertFalse(fig._brain_mask_3d([a,self.pred[0]])[0,0,0])
        affine=np.diag([2.,2.,2.,1.]);affine[2,3]=-72
        self.assertEqual(fig.slice_coordinate(affine,(4,5,6),2,4),'MNI z = -64.0 mm')
        self.assertIn('BrainSurfCNN-like',fig.MODEL_ORDER);self.assertIn('SWIFUN-like',fig.MODEL_ORDER)

if __name__=='__main__':unittest.main()
